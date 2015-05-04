import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats, signal
from sklearn.decomposition import PCA

import nose.tools as nt
import numpy.testing as npt
import pandas.util.testing as pdt

from .. import glm

# Reproducible randomness
rs = np.random.RandomState(sum(map(ord, "glm")))


def test_hrf_sum():
    """Returned HRF values should sum to 1."""
    hrf1 = glm.GammaDifferenceHRF()
    npt.assert_almost_equal(hrf1.kernel.sum(), 1)

    hrf2 = glm.GammaDifferenceHRF(ratio=0)
    npt.assert_almost_equal(hrf2.kernel.sum(), 1)


def test_hrf_peaks():
    """Test HRF based on gamma distribution properties."""
    hrf1 = glm.GammaDifferenceHRF(oversampling=500,
                                  pos_shape=6, pos_scale=1, ratio=0)
    hrf1_peak = hrf1._timepoints[np.argmax(hrf1.kernel)]
    npt.assert_almost_equal(hrf1_peak, 5, 2)

    hrf2 = glm.GammaDifferenceHRF(oversampling=500,
                                  pos_shape=4, pos_scale=2, ratio=0)
    hrf2_peak = hrf2._timepoints[np.argmax(hrf2.kernel)]
    npt.assert_almost_equal(hrf2_peak, (4 - 1) * 2, 2)

    hrf3 = glm.GammaDifferenceHRF(oversampling=500,
                                  neg_shape=7, neg_scale=2, ratio=1000)
    hrf3_trough = hrf3._timepoints[np.argmax(hrf3.kernel)]
    npt.assert_almost_equal(hrf3_trough, (7 - 1) * 2, 2)


def test_hrf_shape():
    """Test the shape of the hrf output with different params."""
    hrf1 = glm.GammaDifferenceHRF()
    npt.assert_equal(hrf1.kernel.shape[1], 1)

    hrf2 = glm.GammaDifferenceHRF(temporal_deriv=True)
    npt.assert_equal(hrf2.kernel.shape[1], 2)


def test_hrf_deriv_scaling():
    """Test relative scaling of main HRF and derivative."""
    hrf = glm.GammaDifferenceHRF(temporal_deriv=True)
    y, dy = hrf.kernel.T
    ss_y = np.square(y).sum()
    ss_dy = np.square(dy).sum()
    npt.assert_almost_equal(ss_y, ss_dy)


def test_hrf_deriv_timing():
    """Test some timing aspects of the HRF and its derivative."""
    hrf = glm.GammaDifferenceHRF(temporal_deriv=True, oversampling=100)
    y, dy = hrf.kernel.T
    nt.assert_greater(np.argmax(y), np.argmax(dy))
    npt.assert_almost_equal(dy[np.argmax(y)], 0, 4)


def test_hrf_convolution():
    """Test some basics about the convolution."""
    hrf = glm.GammaDifferenceHRF()
    data1 = np.zeros(512)
    data1[0] = 1
    conv1 = hrf.convolve(data1)
    npt.assert_almost_equal(float(conv1.sum()), 1)

    data2 = np.ones(512)
    conv2 = hrf.convolve(data2)
    npt.assert_almost_equal(float(conv2.loc[32:].mean()), 1)


def test_hrf_frametimes():
    """Test the frametimes that come out of the convolution."""
    data = (rs.rand(512) < .2).astype(int)
    ft = np.arange(512)

    hrf1 = glm.GammaDifferenceHRF()
    conv1 = hrf1.convolve(data, ft)
    npt.assert_array_equal(ft, conv1.index.values)

    hrf2 = glm.GammaDifferenceHRF(tr=1, oversampling=1)
    conv2 = hrf2.convolve(data)
    npt.assert_array_equal(ft, conv2.index.values)

    hrf3 = glm.GammaDifferenceHRF(tr=2, oversampling=2)
    conv3 = hrf3.convolve(data)
    npt.assert_array_equal(ft, conv3.index.values)


def test_hrf_names():
    """Test the names that come out of the convolution."""
    data = (rs.rand(500) < .2).astype(int)
    series_data = pd.Series(data, name="donna")

    hrf1 = glm.GammaDifferenceHRF()

    conv1 = hrf1.convolve(data)
    nt.assert_equal(conv1.columns.tolist(), ["event"])

    conv2 = hrf1.convolve(data, name="donna")
    nt.assert_equal(conv2.columns.tolist(), ["donna"])

    conv3 = hrf1.convolve(series_data)
    nt.assert_equal(conv3.columns.tolist(), ["donna"])

    hrf2 = glm.GammaDifferenceHRF(temporal_deriv=True)
    conv4 = hrf2.convolve(series_data)
    nt.assert_equal(conv4.columns.tolist(), ["donna", "donna_deriv"])

def test_hrf_impulse_response():

    hrf = glm.GammaDifferenceHRF(tr=1, oversampling=16, kernel_secs=32,
                                 pos_shape=6, pos_scale=1, ratio=0)
    response = hrf.impulse_response

    expected = stats.gamma(6).pdf(hrf._timepoints)
    expected /= expected.sum()
    resampler = sp.interpolate.interp1d(hrf._timepoints,
                                        expected,
                                        "nearest")
    expected = pd.DataFrame(data=resampler(hrf._sampled_timepoints),
                            index=hrf._sampled_timepoints,
                            columns=["Impulse response"])
    pdt.assert_frame_equal(response, expected)


def test_fir_convolution():

    data = np.zeros(144)
    data[::12] = 1

    fir = glm.FIR()
    conv = fir.convolve(data)

    nt.assert_equal(conv.shape, (144, 12))
    npt.assert_array_equal(conv.sum(axis=1), np.ones(144))
    npt.assert_array_equal(conv.values,
                           np.vstack([np.eye(12) for _ in range(12)]))


def test_fir_parametric():

    data = np.zeros(144)
    values = rs.rand(12)
    data[::12] = values

    fir = glm.FIR()
    conv = fir.convolve(data)

    npt.assert_array_equal(conv.sum(axis=1), np.repeat(values, 12))


def test_fir_nbasis():

    data = np.zeros(100)
    data[::12] = 1

    fir1 = glm.FIR(nbasis=10)
    conv1 = fir1.convolve(data)
    nt.assert_equal(conv1.shape, (100, 10))

    fir2 = glm.FIR(nbasis=15)
    conv2 = fir2.convolve(data)
    nt.assert_equal(conv2.shape, (100, 15))


def test_fir_names():

    data = np.zeros(100)
    data[::12] = 1

    fir1 = glm.FIR()
    conv1 = fir1.convolve(data)
    npt.assert_array_equal(conv1.columns,
                           ["event_{:02d}".format(i) for i in range(12)])

    fir2 = glm.FIR()
    conv2 = fir2.convolve(data, name="donna")
    npt.assert_array_equal(conv2.columns,
                           ["donna_{:02d}".format(i) for i in range(12)])


def test_fir_offset():

    data = np.zeros(100)
    data[::12] = 1

    fir1 = glm.FIR(offset=-1)
    conv1 = fir1.convolve(data)
    npt.assert_array_equal(conv1.values[:11], np.eye(12)[1:])
    npt.assert_array_equal(conv1.values[11], np.eye(12)[0])

    fir2 = glm.FIR(offset=1)
    conv2 = fir2.convolve(data)
    npt.assert_array_equal(conv2.values[1:13], np.eye(12))
    npt.assert_array_equal(conv2.values[0], np.zeros(12))


def test_fir_frametimes():

    data = np.zeros(100)
    data[::12] = 1

    fir1 = glm.FIR()
    conv1 = fir1.convolve(data)
    npt.assert_array_equal(conv1.index, np.arange(0, 100 * 2, 2))

    fir2 = glm.FIR(tr=1)
    conv2 = fir2.convolve(data)
    npt.assert_array_equal(conv2.index, np.arange(0, 100))


def test_identity_hrf():
    """Test the identity HRF model."""
    data = rs.randn(20)
    frametimes = np.arange(20)
    name = "donna"

    hrf = glm.IdentityHRF()
    out = hrf.convolve(data, frametimes, name)
    nt.assert_equal(out.columns, [name])
    npt.assert_array_equal(out.index.values, frametimes)
    npt.assert_array_equal(out.donna.values, data)


def test_design_matrix_size():
    """Test the size of the resulting matrix with various options."""
    hrf = glm.GammaDifferenceHRF()
    design = pd.DataFrame(dict(condition=["one", "two"],
                               onset=[5, 20]))
    regressors = rs.randn(20, 2)
    confounds = rs.randn(20, 3)
    artifacts = rs.rand(20) < .1

    X1 = glm.DesignMatrix(design, hrf, 20)
    nt.assert_equal(X1.design_matrix.shape, (20, 2))
    nt.assert_equal(X1.main_submatrix.shape, (20, 2))

    X2 = glm.DesignMatrix(design, hrf, 20, regressors=regressors)
    nt.assert_equal(X2.design_matrix.shape, (20, 4))
    nt.assert_equal(X2.main_submatrix.shape, (20, 4))
    nt.assert_equal(X2.condition_submatrix.shape, (20, 2))

    X3 = glm.DesignMatrix(design, hrf, 20, confounds=confounds)
    nt.assert_equal(X3.design_matrix.shape, (20, 5))
    nt.assert_equal(X3.confound_submatrix.shape, (20, 3))

    X4 = glm.DesignMatrix(design, hrf, 20, regressors=regressors,
                          artifacts=artifacts)
    nt.assert_equal(X4.design_matrix.shape, (20, 4 + artifacts.sum()))
    nt.assert_equal(X2.main_submatrix.shape, (20, 4))
    nt.assert_equal(X4.artifact_submatrix.shape, (20, artifacts.sum()))

    artifacts = np.zeros(20)
    X5 = glm.DesignMatrix(design, hrf, 20, artifacts=artifacts)
    nt.assert_equal(X5.design_matrix.shape, (20, 2))

    hrf2 = glm.GammaDifferenceHRF(temporal_deriv=True)
    X6 = glm.DesignMatrix(design, hrf2, 20, regressors=regressors,
                          confounds=confounds)
    nt.assert_equal(X6.design_matrix.shape, (20, 9))
    nt.assert_equal(X6.main_submatrix.shape, (20, 4))
    nt.assert_equal(X6.condition_submatrix.shape, (20, 2))
    nt.assert_equal(X6.confound_submatrix.shape, (20, 3))


def test_design_matrix_conditions():
    """Test the building of the condition vectors."""
    hrf = glm.IdentityHRF()
    design = pd.DataFrame(dict(condition=["one", "two"],
                          onset=[5, 10]))
    X = glm.DesignMatrix(design, hrf, 15, tr=1, oversampling=1,
                         hpf_cutoff=None)

    one_where = np.asscalar(np.argwhere(X._hires_conditions.one.values))
    nt.assert_equal(one_where, 5)

    two_where = np.asscalar(np.argwhere(X._hires_conditions.two.values))
    nt.assert_equal(two_where, 10)

    x_max = X.design_matrix.idxmax().tolist()
    nt.assert_equal(x_max, [5, 10])


def test_design_matrix_condition_names():
    """Test that we can specify condition names."""
    hrf = glm.IdentityHRF()
    design = pd.DataFrame(dict(condition=["one", "two"],
                          onset=[5, 10]))
    X1 = glm.DesignMatrix(design, hrf, 15)
    nt.assert_equal(X1._condition_names.tolist(), ["one", "two"])

    X2 = glm.DesignMatrix(design, hrf, 15, condition_names=["two", "one"])
    nt.assert_equal(X2._condition_names.tolist(), ["two", "one"])

    X3 = glm.DesignMatrix(design, hrf, 15, condition_names=["one"])
    nt.assert_equal(X3._condition_names.tolist(), ["one"])
    nt.assert_equal(X3.shape, (15, 1))


def test_design_matrix_contrast_vector():
    """Test that we get the right contrast vector under various conditions."""
    hrf = glm.IdentityHRF()
    design = pd.DataFrame(dict(condition=["one", "two"],
                          onset=[5, 10]))
    X1 = glm.DesignMatrix(design, hrf, 15)
    C1 = X1.contrast_vector(["one", "two"], [1, -1])
    nt.assert_equal(C1.tolist(), [1, -1])
    C2 = X1.contrast_vector(["two", "one"], [1, -1])
    nt.assert_equal(C2.tolist(), [-1, 1])

    X2 = glm.DesignMatrix(design, hrf, 15,
                          regressors=rs.randn(15, 1),
                          confounds=rs.randn(15, 1))
    C3 = X2.contrast_vector(["regressor_0"], [1])
    nt.assert_equal(C3.tolist(), [0, 0, 1, 0])


def test_design_matrix_artifacts():
    """Test the creation of artifact regressors."""
    hrf = glm.IdentityHRF()
    design = pd.DataFrame(dict(condition=["one", "two"],
                          onset=[5, 10]))

    X1 = glm.DesignMatrix(design, hrf, 15)
    nt.assert_equal(X1.artifact_submatrix, None)

    artifacts = np.zeros(15, bool)
    X2 = glm.DesignMatrix(design, hrf, 15, artifacts=artifacts)
    nt.assert_equal(X2.artifact_submatrix, None)

    artifacts[10] = True
    X3 = glm.DesignMatrix(design, hrf, 15, artifacts=artifacts)
    nt.assert_equal(X3.artifact_submatrix.idxmax().tolist(), [20])

    artifacts[12] = 1
    X4 = glm.DesignMatrix(design, hrf, 15, artifacts=artifacts, tr=1)
    nt.assert_equal(X4.artifact_submatrix.idxmax().tolist(), [10, 12])

    artifacts[14] = 1.0
    X5 = glm.DesignMatrix(design, hrf, 15, artifacts=artifacts,
                          hpf_cutoff=None)
    art_vals = X5.artifact_submatrix.artifact_2.unique().tolist()
    npt.assert_almost_equal(art_vals, [-1. / 15, 14. / 15])


def test_design_matrix_demeaned():
    """Make sure the design matrix is de-meaned."""
    hrf = glm.GammaDifferenceHRF(temporal_deriv=True)
    design = pd.DataFrame(dict(condition=["one", "two"],
                          onset=[5, 10]))
    artifacts = np.zeros(15, int)
    artifacts[10] = 1
    X = glm.DesignMatrix(design, hrf, 15,
                         regressors=rs.randn(15, 3) + 2,
                         confounds=(rs.randn(15, 3) +
                                    rs.rand(3)),
                         artifacts=artifacts)
    npt.assert_array_almost_equal(X.design_matrix.mean().values,
                                  np.zeros(11))


def test_design_matrix_condition_defaults():
    """Test the condition creation."""
    hrf = glm.GammaDifferenceHRF()
    design1 = pd.DataFrame(dict(condition=["one", "two"],
                                onset=[5, 15]))

    X1 = glm.DesignMatrix(design1, hrf, 20)
    npt.assert_array_equal(X1.design.value.values, np.ones(2))
    npt.assert_array_equal(X1.design.duration.values, np.zeros(2))


def test_design_matrix_frametimes():
    """Test the regular and hires frametimes."""
    hrf = glm.GammaDifferenceHRF(temporal_deriv=True)
    design = pd.DataFrame(dict(condition=["one", "two"],
                          onset=[5, 10]))

    X1 = glm.DesignMatrix(design, hrf, 20, oversampling=2)
    nt.assert_equal(len(X1.frametimes), 20)
    nt.assert_equal(len(X1._hires_frametimes), 40)
    npt.assert_array_equal(X1.frametimes, X1._hires_frametimes[::2])


def test_design_matrix_confound_pca():
    """Test the PCA transformation of the confound matrix."""
    hrf = glm.GammaDifferenceHRF(temporal_deriv=True)
    design = pd.DataFrame(dict(condition=["one", "two"], onset=[5, 10]))
    confounds = rs.randn(20, 5)
    confounds[:, 0] = confounds[:, 1] + rs.randn(20)
    transformed_confounds = PCA(.99).fit_transform(confounds)
    X = glm.DesignMatrix(design, hrf, 20,
                         confounds=confounds,
                         confound_pca=True)
    n_confounds = X.confound_submatrix.shape[1]
    nt.assert_equal(n_confounds, transformed_confounds.shape[1])

    pca_all = PCA().fit(confounds)
    good_dims = np.sum(pca_all.explained_variance_ratio_ > .01)
    nt.assert_equal(n_confounds, good_dims)


def test_design_matrix_precomputed_kernel():
    """Test that we can use a precomputed highpass filter matrix."""
    F = glm.fsl_highpass_matrix(20, 8, 2)
    hrf = glm.GammaDifferenceHRF()

    design = pd.DataFrame(dict(condition=["one", "two"],
                               onset=[5, 20]))
    regressors = rs.randn(20, 2)

    X_1 = glm.DesignMatrix(design, hrf, 20, regressors=regressors, hpf_cutoff=8)
    X_2 = glm.DesignMatrix(design, hrf, 20, regressors=regressors, hpf_kernel=F)

    npt.assert_array_equal(X_1.design_matrix, X_2.design_matrix)


def test_design_matrix_only_regressors():
    """Test that the design component of the matrix is optional."""
    regressors = rs.randn(20, 2)
    design = glm.DesignMatrix(regressors=regressors)
    npt.assert_array_equal(design.design_matrix.columns,
                           ["regressor_0", "regressor_1"])


def test_design_matrix_only_confounds():
    """Test that the design component of the matrix is optional."""
    confounds = rs.randn(20, 2)
    design = glm.DesignMatrix(confounds=confounds)
    npt.assert_array_equal(design.design_matrix.columns,
                           ["confound_0", "confound_1"])


def test_design_matrix_with_fir():

    design = pd.DataFrame(dict(onset=np.arange(0, 100, 10),
                               condition=["event"] * 10))

    X1 = glm.DesignMatrix(design, glm.FIR(tr=1, offset=0), ntp=100,
                          tr=1, oversampling=1)
    nt.assert_equal(X1.shape, (100, 12))

    design["condition"] = np.tile(["event", "other_event"], 5)    
    X2 = glm.DesignMatrix(design, glm.FIR(tr=1, offset=0), ntp=100,
                          tr=1, oversampling=1)
    nt.assert_equal(X2.shape, (100, 24))
    

def test_highpass_matrix_shape():
    """Test the filter matrix is the right shape."""
    for n_tp in 10, 100:
        F = glm.fsl_highpass_matrix(n_tp, 50)
        nt.assert_equal(F.shape, (n_tp, n_tp))


def test_filter_matrix_diagonal():
    """Test that the filter matrix has strong diagonal."""
    F = glm.fsl_highpass_matrix(10, 3)
    npt.assert_array_equal(F.argmax(axis=1).squeeze(), np.arange(10))


def test_filtered_data_shape():
    """Test that filtering data returns same shape."""
    data = rs.randn(100)
    data_filt = glm.fsl_highpass_filter(data, 30)
    nt.assert_equal(data.shape, data_filt.shape)

    data = rs.randn(100, 3)
    data_filt = glm.fsl_highpass_filter(data, 30)
    nt.assert_equal(data.shape, data_filt.shape)


def test_filter_psd():
    """Test highpass filter with power spectral density."""
    a = np.sin(np.linspace(0, 4 * np.pi, 32))
    b = rs.randn(32) / 2
    y = a + b
    y_filt = glm.fsl_highpass_filter(y, 10)
    nt.assert_equal(y.shape, y_filt.shape)

    _, orig_d = signal.welch(y, nperseg=32)
    _, filt_d = signal.welch(y_filt, nperseg=32)

    nt.assert_greater(orig_d.sum(), filt_d.sum())


def test_filter_strength():
    """Test that lower cutoff makes filter more aggresive."""
    a = np.sin(np.linspace(0, 4 * np.pi, 32))
    b = rs.randn(32) / 2
    y = a + b

    cutoffs = np.linspace(20, 80, 5)
    densities = np.zeros_like(cutoffs)
    for i, cutoff in enumerate(cutoffs):
        filt = glm.fsl_highpass_filter(y, cutoff)
        _, density = signal.welch(filt, nperseg=32)
        densities[i] = density.sum()

    npt.assert_array_equal(densities, np.sort(densities))


def test_filter_copy():
    """Test that copy argument to filter function works."""
    a = rs.randn(100, 10)
    a_copy = glm.fsl_highpass_filter(a, 50, copy=True)
    assert(not (a == a_copy).all())
    a_nocopy = glm.fsl_highpass_filter(a, 100, copy=False)
    npt.assert_array_equal(a, a_nocopy)
