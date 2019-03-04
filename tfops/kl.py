import tensorflow as tf

def _squared_frobenius_norm(x):
    """Helper to make KL calculation slightly more readable."""
    # http://mathworld.wolfram.com/FrobeniusNorm.html
    return tf.square(tf.norm(x, ord="fro", axis=[-2, -1]))

def _multivariate_kl_sym(posterior_means, posterior_covariance,
                         prior_mean, prior_covariance,
                         batch_size, embedding_dimension):
    """Compute symbolic kl divergence given posterior and prior multivariate
    Gaussian parameters. KL(a || b).
    KL(a || b) = 0.5 * ( L - k + T + Q ),
    L := Log[Det(C_b)] - Log[Det(C_a)]
    T := trace(C_b^{-1} C_a),
    Q := (mu_b - mu_a)^T C_b^{-1} (mu_b - mu_a),
    """

    a_scale = tf.contrib.linalg.LinearOperatorTriL(tf.cholesky(posterior_covariance))
    b_scale = tf.contrib.linalg.LinearOperatorTriL(tf.cholesky(prior_covariance))

    b_inv_a = b_scale.solve(a_scale.to_dense())
    # First compute everything but Q
    kl_div_L = b_scale.log_abs_determinant() - a_scale.log_abs_determinant()
    kl_div_L = tf.Print(kl_div_L, ['kl_logdet: ', kl_div_L])
    kl_div_T = 0.5 * (-1.0 * embedding_dimension
                      + _squared_frobenius_norm(b_inv_a))
    kl_div_T = tf.Print(kl_div_T, ['kl_trace: ', kl_div_T])

    Qs = b_scale.solve(tf.transpose(prior_mean - posterior_means))
    #   kl_Qs = tf.reduce_sum(tf.square(Qs))
    kl_div_Qs = tf.reduce_mean(tf.reduce_sum(tf.square(Qs), 0))
    kl_div_Qs = tf.Print(kl_div_Qs, ['kl_Q: ', kl_div_Qs])

    #   kl_div = batch_size * kl_div + 0.5 * kl_Qs
    kl_div = kl_div_L + kl_div_T + 0.5 * kl_div_Qs
    kl_div = tf.Print(kl_div, ['Final kl: ', kl_div])
    return kl_div


