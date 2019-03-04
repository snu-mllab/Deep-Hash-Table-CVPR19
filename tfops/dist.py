import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.gpu_op import selectGpuById
from tfops.info_op import get_shape

from tensorflow.python.framework import dtypes, ops, sparse_tensor, tensor_shape
from tensorflow.python.ops import array_ops, control_flow_ops, logging_ops, math_ops,\
                                  nn, script_ops, sparse_ops
from tensorflow.python.summary import summary
import tensorflow as tf

def deviation_from_kth_element(feature, k):
    '''
    Args:
        feature - 2-D tensor of size [number of data, feature dimensions]
        k - int
            number of bits to be activated
            k < feature dimensions
    Return:
        deviation : 2-D Tensor of size [number of data, feature dimensions] 
           deviation from k th element 
    '''
    feature_top_k = tf.nn.top_k(feature, k=k+1)[0] # [number of data, k+1]

    rho = tf.stop_gradient(tf.add(feature_top_k[:,k-1], feature_top_k[:,k])/2) # [number of data]
    rho_r = tf.reshape(rho, [-1,1]) # [number of data, 1] 

    deviation = tf.subtract(feature, rho_r) # [number of data, feature dimensions]
    return deviation


def pairwise_distance_euclid_v2(feature1, feature2):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
    	feature1 - 2-D Tensor of size [number of data1, feature dimension].
    	feature2 - 2-D Tensor of size [number of data2, feature dimension].
    Returns:
    	pairwise_distances - 2-D Tensor of size [number of data1, number of data2].
    """
    pairwise_distances = math_ops.add(
    	math_ops.reduce_sum(
	    math_ops.square(feature1),
	    axis=[1],
	    keep_dims=True),
        math_ops.reduce_sum(
	    math_ops.square(
	        array_ops.transpose(feature2)),
	        axis=[0],
	        keep_dims=True)) - 2.0 * math_ops.matmul(
	    feature1, array_ops.transpose(feature2))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances = math_ops.maximum(pairwise_distances, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances, 0.0)

    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))
    return pairwise_distances

def pairwise_distance_euclid(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
    	feature - 2-D Tensor of size [number of data, feature dimension].
    	squared - Boolean, whether or not to square the pairwise distances.
    Returns:
    	pairwise_distances - 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
    	math_ops.reduce_sum(
	    math_ops.square(feature),
	    axis=[1],
	    keep_dims=True),
        math_ops.reduce_sum(
	    math_ops.square(
	        array_ops.transpose(feature)),
	        axis=[0],
	        keep_dims=True)) - 2.0 * math_ops.matmul(
	    feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
        data : float, 2D tensor [n, m]
        mask : bool, 2D tensor [n, m]
        dim : int, the dimension over which to compute the maximum.
    Returns:
        masked_maximums : 2D Tensor
            dim=0 => [n,1]
            dim=1 => [1,m]
        get maximum among mask=1
    """
    axis_minimums = math_ops.reduce_min(data, dim, keep_dims=True)
    masked_maximums = math_ops.reduce_max(math_ops.multiply(data - axis_minimums, mask), dim, keep_dims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
        data : float, 2D tensor [n, m]
        mask : bool, 2D tensor [n, m]
        dim : int, the dimension over which to compute the minimum.
    Returns:
        masked_minimums : 2D Tensor
            dim=0 => [n,1]
            dim=1 => [1,m]
        get minimum among mask=1
    """
    axis_maximums = math_ops.reduce_max(data, dim, keep_dims=True) # [n, 1] or [1, m]
    masked_minimums = math_ops.reduce_min(math_ops.multiply(data - axis_maximums, mask), dim, keep_dims=True) + axis_maximums # [n, 1] or [1, m]
    return masked_minimums

def triplet_semihard_loss(labels, embeddings, pairwise_distance, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
    Args:
	labels - 1-D tensor [batch_size] as tf.int32
                multiclass integer labels.
	embeddings - 2-D tensor [batch_size, feature dimensions] as tf.float32
                     `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        pairwise_distance - func 
                            with argus 2D tensor [number of data, feature dims]
                                 return 2D tensor [number of data, number of data] 
	margin - float defaults to be 1.0
                margin term in the loss definition.

    Returns:
    	triplet_loss - tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # pdist_matrix[i][j] = dist(i, j)
    pdist_matrix = pairwise_distance(embeddings) # [batch_size, batch_size]
    # adjacency[i][j]=1 if label[i]==label[j] else 0
    adjacency = math_ops.equal(labels, array_ops.transpose(labels)) # [batch_size, batch_size]
    # adjacency_not[i][j]=0 if label[i]==label[j] else 0
    adjacency_not = math_ops.logical_not(adjacency) # [batch_size, batch_size]
    batch_size = array_ops.size(labels)

    # Compute the mask.
    # pdist_matrix_tile[batch_size*i+j, k] = distance(j, k)
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1]) # [batch_size*batch_size, batch_size]
    # reshape(transpose(pdist_matrix), [-1, 1])[batch_size*i+j][0] = distance(j, i)
    # tile(adjacency_not, [batch_size, 1])[batch_size*i+j][k] = 1 if label[j]!=label[k] otherwise 0
    # mask[batch_size*i+j][k] = 1 if label[j]!=label[k] different label, and distance(j,k)>distance(j,i)
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(pdist_matrix_tile, array_ops.reshape(array_ops.transpose(pdist_matrix), [-1, 1]))) # [batch_size*batch_size, batch_size]
    # mask_final[i][j]=1 if there exists k s.t. label[j]!=label[k] and distance(j,k)>distance(j,i)
    mask_final = array_ops.reshape(
        math_ops.greater(math_ops.reduce_sum(math_ops.cast(mask, dtype=dtypes.float32), 1, keep_dims=True), 0.0),
        [batch_size, batch_size])# [batch_size, batch_size]
    # mask_final[i][j]=1 if there exists k s.t. label[i]!=label[k] and distance(i,k)>distance(i,j)
    mask_final = array_ops.transpose(mask_final)# [batch_size, batch_size]

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32) # [batch_size, batch_size]
    mask = math_ops.cast(mask, dtype=dtypes.float32) # [batch_size*batch_size, batch_size]

    # masked_minimum(pdist_matrix_tile, mask)[batch*i+j][1] = pdist_matrix[j][k] s.t minimum over 'k's label[j]!=label[k], distance(j,k)>distance(j,i)
    # negatives_outside[i][j] = pdist_matrix[j][k] s.t minimum over 'k's label[j]!=label[k], distance(j,k)>distance(j,i)
    negatives_outside = array_ops.reshape(masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]) # [batch_size, batch_size]
    # negatives_outside[i][j] = pdist_matrix[i][k] s.t minimum over 'k's label[i]!=label[k], distance(i,k)>distance(i,j)
    negatives_outside = array_ops.transpose(negatives_outside) # [batch_size, batch_size]

    # negatives_inside[i][j] = pdist_matrix[i][k] s.t maximum over label[i]!=label[k]
    negatives_inside = array_ops.tile(masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]) # [batch_size, batch_size]
     
    # semi_hard_negatives[i][j] = pdist_matrix[i][k] if exists negatives_outside, otherwise negatives_inside
    semi_hard_negatives = array_ops.where(mask_final, negatives_outside, negatives_inside) # [batch_size, batch_size]
    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives) # [batch_size, batch_size]

    # mask_positives[i][j] = 1 if label[i]==label[j], and i!=j
    mask_positives = math_ops.cast(adjacency, dtype=dtypes.float32) - array_ops.diag(array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    # in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    triplet_loss = math_ops.truediv(
            math_ops.reduce_sum(math_ops.maximum(math_ops.multiply(loss_mat, mask_positives), 0.0)),
            num_positives,
            name='triplet_semihard_loss') # hinge
    return triplet_loss

def triplet_semihard_loss_selective(labels, plabels, embeddings, pairwise_distance, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
    Args:
	labels - 1-D tensor [batch_size] as tf.int32
                multiclass integer labels.
	plabels - 1-D tensor [batch_size] as tf.int32
                to be selective previous label
	embeddings - 2-D tensor [batch_size, feature dimensions] as tf.float32
                     `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        pairwise_distance - func 
                            with argus 2D tensor [number of data, feature dims]
                                 return 2D tensor [number of data, number of data] 
	margin - float defaults to be 1.0
                margin term in the loss definition.

    Returns:
    	triplet_loss - tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    batch_size = array_ops.size(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1]) # [batch_size, 1]
    plabels = array_ops.reshape(plabels, [lshape[0], 1]) # [batch_size, 1]
    # pdist_matrix[i][j] = dist(i, j)
    pdist_matrix = pairwise_distance(embeddings) # [batch_size, batch_size]
    # padjacency[i][j]=1 if plabels[i]==plabels[j] else 0
    padjacency = math_ops.equal(plabels, array_ops.transpose(plabels)) # [batch_size, batch_size]
    # adjacency_tmp[i][j]=1 if labels[i]==labels[j] else 0
    adjacency_tmp = math_ops.equal(plabels, array_ops.transpose(plabels)) # [batch_size, batch_size]
    # adjacency[i][j]=1 if plabels[i]==plabels[j], labels[i]==labels[j] else 0
    adjacency = math_ops.logical_and(adjacency_tmp, padjacency) # [batch_size, batch_size]
    # adjacency_not[i][j]=1 if plabels[i]==plabels[j], labels[i]!=labels[j] else 0
    adjacency_not = math_ops.logical_and(math_ops.logical_not(adjacency_tmp), padjacency) # [batch_size, batch_size]
    # Compute the mask.
    # pdist_matrix_tile[batch_size*i+j, k] = distance(j, k)
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1]) # [batch_size*batch_size, batch_size]
    # reshape(transpose(pdist_matrix), [-1, 1])[batch_size*i+j][0] = distance(j, i)
    # tile(adjacency_not, [batch_size, 1])[batch_size*i+j][k] = 1 if plabels[j]==plabels[k] and labels[j]!=labels[k] otherwise 0
    # mask[batch_size*i+j][k] = 1 if plabels[j]==plabels[k] and labels[j]!=labels[k], and distance(j,k)>distance(j,i)
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(pdist_matrix_tile, array_ops.reshape(array_ops.transpose(pdist_matrix), [-1, 1]))) # [batch_size*batch_size, batch_size]
    # mask_final[i][j]=1 if there exists k s.t. plabels[j]==plabels[k] and labels[j]!=labels[k] and distance(j,k)>distance(j,i)
    mask_final = array_ops.reshape(
        math_ops.greater(math_ops.reduce_sum(math_ops.cast(mask, dtype=dtypes.float32), 1, keep_dims=True), 0.0),
        [batch_size, batch_size])# [batch_size, batch_size]
    # mask_final[i][j]=1 if there exists k s.t. plabels[i]==plabels[k], labels[i]!=labels[k] and distance(i,k)>distance(i,j)
    mask_final = array_ops.transpose(mask_final)# [batch_size, batch_size]

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32) # [batch_size, batch_size]
    mask = math_ops.cast(mask, dtype=dtypes.float32) # [batch_size*batch_size, batch_size]

    # masked_minimum(pdist_matrix_tile, mask)[batch*i+j][1] = pdist_matrix[j][k] s.t minimum over 'k's plabels[j]==plabels[k], labels[j]!=labels[k], distance(j,k)>distance(j,i)
    # negatives_outside[i][j] = pdist_matrix[j][k] s.t minimum over 'k's plabels[j]==plabels[k], labels[j]!=labels[k], distance(j,k)>distance(j,i)
    negatives_outside = array_ops.reshape(masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]) # [batch_size, batch_size]
    # negatives_outside[i][j] = pdist_matrix[i][k] s.t minimum over 'k's plabels[j]==plabels[k], labels[i]!=labels[k], distance(i,k)>distance(i,j)
    negatives_outside = array_ops.transpose(negatives_outside) # [batch_size, batch_size]

    # negatives_inside[i][j] = pdist_matrix[i][k] s.t maximum over plabels[i]==plabels[k] and labels[i]!=labels[k]
    negatives_inside = array_ops.tile(masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]) # [batch_size, batch_size]
     
    # semi_hard_negatives[i][j] = pdist_matrix[i][k] if exists negatives_outside, otherwise negatives_inside
    semi_hard_negatives = array_ops.where(mask_final, negatives_outside, negatives_inside) # [batch_size, batch_size]
    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives) # [batch_size, batch_size]

    # mask_positives[i][j] = 1 if plabels[i]==plabels[j] and labels[i]==labels[j], and i!=j
    mask_positives = math_ops.cast(adjacency, dtype=dtypes.float32) - array_ops.diag(array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    # in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    triplet_loss = math_ops.truediv(
            math_ops.reduce_sum(math_ops.maximum(math_ops.multiply(loss_mat, mask_positives), 0.0)),
            num_positives,
            name='triplet_semihard_loss') # hinge
    return triplet_loss

def custom_npairs_loss(labels, embeddings_anchor, embeddings_positive, pairwise_similarity, reg_lambda=0.002, print_losses=False):
    """Custom similarity Computes the npairs loss.

    Args:
        labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
        embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the anchor images. Embeddings should not be
            l2 normalized.
        embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the positive images. Embeddings should not be
            l2 normalized.
        pairwise_similarity - func
        reg_lambda: Float. L2 regularization term on the embedding vectors.
        print_losses: Boolean. Option to print the xent and l2loss.
    Returns:
        npairs_loss: tf.float32 scalar.
    """
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    similarity_matrix = pairwise_similarity(embeddings_anchor, embeddings_positive)

    # Reshape [batch_size/2] label tensor to a [batch_size/2, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # labels_remapped[i][j] = 1 if label[i] == label[j] otherwise 0
    labels_remapped = math_ops.to_float(math_ops.equal(labels, array_ops.transpose(labels))) # [batch_size/2, batch_size/2] 
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
        xent_loss = logging_ops.Print(xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

    return l2loss + xent_loss

def npairs_loss(labels, embeddings_anchor, embeddings_positive,
                reg_lambda=0.002, print_losses=False):
    """Computes the npairs loss.
    Npairs loss expects paired data where a pair is composed of samples from the
    same labels and each pairs in the minibatch have different labels. The loss
    has two components. The first component is the L2 regularizer on the
    embedding vectors. The second component is the sum of cross entropy loss
    which takes each row of the pair-wise similarity matrix as logits and
    the remapped one-hot labels as labels.
    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
        labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
        embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the anchor images. Embeddings should not be
            l2 normalized.
        embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the positive images. Embeddings should not be
            l2 normalized.
        reg_lambda: Float. L2 regularization term on the embedding vectors.
        print_losses: Boolean. Option to print the xent and l2loss.
    Returns:
        npairs_loss: tf.float32 scalar.
    """
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    similarity_matrix = math_ops.matmul(embeddings_anchor, embeddings_positive, transpose_a=False, transpose_b=True) # [batch_size/2, batch_size/2]

    # Reshape [batch_size/2] label tensor to a [batch_size/2, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # labels_remapped[i][j] = 1 if label[i] == label[j] otherwise 0
    labels_remapped = math_ops.to_float(math_ops.equal(labels, array_ops.transpose(labels))) # [batch_size/2, batch_size/2] 
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
        xent_loss = logging_ops.Print(xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

    return l2loss + xent_loss

def npairs_loss_selective(labels, plabels, embeddings_anchor, embeddings_positive, reg_lambda=0.002):
    """Computes the npairs loss.
    Args:
        labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
        plabels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
        embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the anchor images. Embeddings should not be
            l2 normalized.
        embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the positive images. Embeddings should not be
            l2 normalized.
        reg_lambda: Float. L2 regularization term on the embedding vectors.
    Returns:
        npairs_loss: tf.float32 scalar.
    """
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    lshape = array_ops.shape(labels)
    assert lshape.shape == 1

    plabels = array_ops.reshape(plabels, [lshape[0], 1]) # [batch_size/2, 1]
    labels = array_ops.reshape(labels, [lshape[0], 1]) # [batch_size/2, 1]

    # padjacency[i][j]=1 if plabels[i]==plabels[j] else 0
    padjacency = math_ops.equal(plabels, array_ops.transpose(plabels)) # [batch_size/2, batch_size/2]
    # adjacency_tmp[i][j]=1 if labels[i]==labels[j] else 0
    adjacency_tmp = math_ops.equal(labels, array_ops.transpose(labels)) # [batch_size/2, batch_size/2]
    # adjacency[i][j]=1 if plabel[i]==plabel[j], labels[i]==labels[j] else 0
    adjacency = math_ops.to_float(math_ops.logical_and(adjacency_tmp, padjacency)) # [batch_size/2, batch_size/2]
    adjacency /= math_ops.reduce_sum(adjacency, 1, keep_dims=True) # [batch_size/2, batch_size/2]
    # Get per pair similarities.
    similarity_matrix = math_ops.matmul(embeddings_anchor, embeddings_positive, transpose_a=False, transpose_b=True) # [batch_size/2, batch_size/2]
    similarity_softmax = tf.multiply(math_ops.to_float(padjacency), tf.exp(similarity_matrix)) # [batch_size/2, batch_size/2]
    similarity_softmax = tf.exp(similarity_matrix)/math_ops.reduce_sum(similarity_softmax, axis=1, keep_dims=True) # [batch_size/2, batch_size/2]
    # calculating cross_entropy losses
    xent_loss = tf.negative(tf.multiply(adjacency, tf.multiply(math_ops.to_float(padjacency), tf.log(tf.clip_by_value(similarity_softmax, 1e-7, 1.0)))))
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')
    return l2loss + xent_loss

def _build_multilabel_adjacency(sparse_labels):
    """Builds multilabel adjacency matrix.
    As of March 14th, 2017, there's no op for the dot product between
    two sparse tensors in TF. However, there is `sparse_minimum` op which is
    equivalent to an AND op between two sparse boolean tensors.
    This computes the dot product between two sparse boolean inputs.
    Args:
        sparse_labels: List of 1-D boolean sparse tensors.
    Returns:
        adjacency_matrix: 2-D dense `Tensor`.
    """
    num_pairs = len(sparse_labels)
    adjacency_matrix = array_ops.zeros([num_pairs, num_pairs])
    for i in range(num_pairs):
        for j in range(num_pairs):
            sparse_dot_product = math_ops.to_float(
              sparse_ops.sparse_reduce_sum(sparse_ops.sparse_minimum(
                  sparse_labels[i], sparse_labels[j])))
            sparse_dot_product = array_ops.expand_dims(sparse_dot_product, 0)
            sparse_dot_product = array_ops.expand_dims(sparse_dot_product, 1)
            one_hot_matrix = array_ops.pad(sparse_dot_product,
                                         [[i, num_pairs-i-1],
                                          [j, num_pairs-j-1]], 'CONSTANT')
            adjacency_matrix += one_hot_matrix

    return adjacency_matrix

def npairs_loss_multilabel(sparse_labels, embeddings_anchor,
                           embeddings_positive, reg_lambda=0.002,
                           print_losses=False):
    """Computes the npairs loss with multilabel data.
    Npairs loss expects paired data where a pair is composed of samples from the
    same labels and each pairs in the minibatch have different labels. The loss
    has two components. The first component is the L2 regularizer on the
    embedding vectors. The second component is the sum of cross entropy loss
    which takes each row of the pair-wise similarity matrix as logits and
    the remapped one-hot labels as labels. Here, the similarity is defined by the
    dot product between two embedding vectors. S_{i,j} = f(x_i)^T f(x_j)
    To deal with multilabel inputs, we use the count of label intersection
    i.e. L_{i,j} = | set_of_labels_for(i) \cap set_of_labels_for(j) |
    Then we normalize each rows of the count based label matrix so that each row
    sums to one.
    Args:
        sparse_labels: List of 1-D Boolean `SparseTensor` of dense_shape
                       [batch_size/2, num_classes] labels for the anchor-pos pairs.
        embeddings_anchor: 2-D `Tensor` of shape [batch_size/2, embedding_dim] for
          the embedding vectors for the anchor images. Embeddings should not be
          l2 normalized.
        embeddings_positive: 2-D `Tensor` of shape [batch_size/2, embedding_dim] for
          the embedding vectors for the positive images. Embeddings should not be
          l2 normalized.
        reg_lambda: Float. L2 regularization term on the embedding vectors.
        print_losses: Boolean. Option to print the xent and l2loss.
    Returns:
        npairs_loss: tf.float32 scalar.
    Raises:
        TypeError: When the specified sparse_labels is not a `SparseTensor`.
    """

    if False in [isinstance(
        l, sparse_tensor.SparseTensor) for l in sparse_labels]:
        raise TypeError(
            'sparse_labels must be a list of SparseTensors, but got %s' % str(
                sparse_labels))

    with ops.name_scope('NpairsLossMultiLabel'):
        # Add the regularizer on the embedding.
        reg_anchor = math_ops.reduce_mean(
            math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
        reg_positive = math_ops.reduce_mean(
            math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
        l2loss = math_ops.multiply(0.25 * reg_lambda,
                                   reg_anchor + reg_positive, name='l2loss')

        # Get per pair similarities.
        similarity_matrix = math_ops.matmul(
            embeddings_anchor, embeddings_positive, transpose_a=False,
            transpose_b=True)

        # TODO(coreylynch): need to check the sparse values
        # TODO(coreylynch): are composed only of 0's and 1's.

        multilabel_adjacency_matrix = _build_multilabel_adjacency(sparse_labels)
        labels_remapped = math_ops.to_float(multilabel_adjacency_matrix)
        labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

        # Add the softmax loss.
        xent_loss = nn.softmax_cross_entropy_with_logits(
            logits=similarity_matrix, labels=labels_remapped)
        xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

        if print_losses:
            xent_loss = logging_ops.Print(xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

    return l2loss + xent_loss

def lifted_struct_loss(labels, embeddings, margin=1.0):
    """Computes the lifted structured loss.
    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than any negative distances (between a
    pair of embeddings with different labels) in the mini-batch in a way
    that is differentiable with respect to the embedding vectors.
    See: https://arxiv.org/abs/1511.06452.
    Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should not
          be l2 normalized.
        margin: Float, margin term in the loss definition.
    Returns:
        lifted_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pairwise_distances = pairwise_distance(embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    diff = margin - pairwise_distances
    mask = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    # Safe maximum: Temporarily shift negative distances
    #   above zero before taking max.
    #     this is to take the max only among negatives.
    row_minimums = math_ops.reduce_min(diff, 1, keep_dims=True)
    row_negative_maximums = math_ops.reduce_max(
      math_ops.multiply(
          diff - row_minimums, mask), 1, keep_dims=True) + row_minimums

    # Compute the loss.
    # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
    #   where m_i is the max of alpha - negative D_i's.
    # This matches the Caffe loss layer implementation at:
    #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp  # pylint: disable=line-too-long

    max_elements = math_ops.maximum(
      row_negative_maximums, array_ops.transpose(row_negative_maximums))
    diff_tiled = array_ops.tile(diff, [batch_size, 1])
    mask_tiled = array_ops.tile(mask, [batch_size, 1])
    max_elements_vect = array_ops.reshape(
      array_ops.transpose(max_elements), [-1, 1])

    loss_exp_left = array_ops.reshape(
      math_ops.reduce_sum(math_ops.multiply(
          math_ops.exp(
              diff_tiled - max_elements_vect),
          mask_tiled), 1, keep_dims=True), [batch_size, batch_size])

    loss_mat = max_elements + math_ops.log(
      loss_exp_left + array_ops.transpose(loss_exp_left))
    # Add the positive distance.
    loss_mat += pairwise_distances

    mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

    # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
    num_positives = math_ops.reduce_sum(mask_positives) / 2.0

    lifted_loss = math_ops.truediv(
          0.25 * math_ops.reduce_sum(
              math_ops.square(
                  math_ops.maximum(
                      math_ops.multiply(loss_mat, mask_positives), 0.0))),
          num_positives,
          name='liftedstruct_loss')
    return lifted_loss

if __name__ == '__main__':
    # selectGpuById(0)
    def test2():
        sess=tf.Session()
        feature1 = tf.constant([[1,2,3], [-4,-2,0], [5,-1,1], [3,-1,1]], dtype=tf.float32) 
        feature2 = tf.constant([[1,2,3], [-4,-2,0]], dtype=tf.float32) 
        print("feature1 : ", sess.run(feature1))
        print("feature2 : ", sess.run(feature2))
        print(sess.run(pairwise_distance_euclid_v2(feature1, feature2)))
        print(sess.run(pairwise_distance_euclid(feature1, squared=True)))

    test2()
