import numpy as np


TARGET_PSI1 = "psi_1"
TARGET_DPSI_UP = "delta_psi_up"
TARGET_DPSI_DOWN = "delta_psi_down"
NO_OF_POINTS = 250
NO_BASELINES = 3
LOW_KEY = '0'
HIGH_KEY = '1'
BASELINES_ZERO = "zero"
BASELINES_ENCODER_ZERO = "encoded_zero"
BASELINES_K_MEANS = "k-means"
BASELINES_MEDIAN = "median"
BASELINES_RANDOM = "random"
MAX_POINTS_FOR_K_MEANS = 20000
SAMPLE_TYPE_INC = "_inc_idx"
SAMPLE_TYPE_EXC = "_exc_idx"
SAMPLE_TYPE_NC = "nc_change"
MAX_SAMPLES = 800
TISSUE_PAIRS = ['Heart_Spleen', 'Heart_Hippocampus', 'Hippocampus_Liver', 'Hippocampus_Spleen', 'Hippocampus_Lung',
                'Heart_Thymus', 'Lung_Spleen', 'Heart_Liver', 'Spleen_Thymus', 'Liver_Spleen',
                'Liver_Lung', 'Liver_Thymus', 'Hippocampus_Thymus', 'Heart_Lung', 'Lung_Thymus']
TISSUES = ["Heart", "Hippocampus", "Liver", "Lung", "Spleen", "Thymus"]


def get_meta_feature_groups(feature_names, meta_features_file):
    """
    Given feature names, return the indices of members of all meta features.
    :param feature_names: np.array, list of feature names
    :param meta_features_file: str, file containing list of feature names and meta-feature names
    :return: list of subgroup indices and meta-feature names
    """
    meta_features = np.loadtxt(meta_features_file, dtype=str)
    meta_feat_names = meta_features[:, 3]
    feature_mappings = meta_features[:, 2]

    meta_feature_dict = {}
    for ii, jj in zip(meta_feat_names, feature_mappings):
        if jj in feature_names:
            idx = list(np.where(feature_names == jj)[0])
            if ii not in meta_feature_dict.keys():
                meta_feature_dict[ii] = idx
            else:

                meta_feature_dict[ii].extend(idx)

    for ii in list(meta_feature_dict.keys()):
        meta_feature_dict[ii] = np.array(meta_feature_dict[ii], dtype=int)
    meta_feat_names, meta_feat_subgroups = zip(*meta_feature_dict.items())
    return np.array(meta_feat_names, dtype=str), meta_feat_subgroups


def get_tissue_names(pair_ids):
    """
    Given tissue pair indices, return names of tissues being compared.
    :param pair_ids: list of tissue pair idx from TISSUE_PAIRS list
    :return:
    """
    pair_strings = [TISSUE_PAIRS[pair_id] for pair_id in pair_ids.astype(int)]
    t_pairs = [pair_str.split("_") for pair_str in pair_strings]
    return np.array([[t1, t2] for (t1, t2) in t_pairs])


def get_reverse_corrected_labels(labels,
                                 tissue_set_1,
                                 tissue_set_2,
                                 relevant_idx_reverse):
    """
    Since the tissue of interest might be tissue 1 or tissue 2 in the deltaPSI comparison, flip the deltaPSI value
    and tissue ids if the tissue of interest is the tissue 2.
    :param labels: np.array, tissue comparison labels containing PSI1, PSI2 and DeltaPSI
    :param tissue_set_1: np.array, tissue 1 hot vector encoding
    :param tissue_set_2: np.array, tissue 2 hot vector encoding
    :param relevant_idx_reverse: np.array, which indices contain tissue of interest as flipped
    :return: np.array, labels with flipped labels at relevant_idx_reverse indices, np.array, tissue_set_1 with tissue
    one hot vector encoding from tissue_set_2 at relevant_idx_reverse indices, np.array, tissue_set_2 with tissue
    one hot vector encoding from tissue_set_1 at relevant_idx_reverse indices.
    """
    # Save PSI1 to be reversed in tmp
    tmp = labels[relevant_idx_reverse, 0]
    # Put PSI2 to PSI1 at indices where tissue has to be flipped
    labels[relevant_idx_reverse, 0] = labels[relevant_idx_reverse, 1]
    # Put PSI1 into PSI2 at indices where tissue has to be flipped
    labels[relevant_idx_reverse, 1] = tmp
    # Flip the deltaPSI labels for indices where tissue has to be flipped
    labels[relevant_idx_reverse, 2] = -1.0 * labels[relevant_idx_reverse, 2]
    # Put tissue-1 hot vector to tmp
    tmp = tissue_set_1[relevant_idx_reverse, :]
    # Put tissue-2 hot vector to tissue-1 hot vector at indices where tissue has to be flipped
    tissue_set_1[relevant_idx_reverse, :] = tissue_set_2[relevant_idx_reverse, :]
    # Put tissue-1 hot vector to tissue-2 hot vector at indices where tissue has to be flipped
    tissue_set_2[relevant_idx_reverse, :] = tmp
    return labels, tissue_set_1, tissue_set_2


def load_data_alt_const(input_data_file):
    """
    This function takes saved npz models for alternative tissue specific data and AS versus Const data and returns
    dictionaries
    :param input_data_file: str, .npz file containing alternative splicing_old data
    :return: a dictionary containing alternative splicing_old and alternative versus constitutive data
    """
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    input_data = np.load(input_data_file, encoding='latin1')
    #np.load.__defaults__=(None, False, True, 'ASCII')
    # restore np.load for future normal usage
    np.load = np_load_old

    # load alternative data
    alt_data = input_data['alt_data'][()]
    # load normalized features for nearest neighbor graph
    features = input_data['norm_features']
    # load constitutive data
    const_data = input_data['const_data'][()]
    return alt_data, features, const_data


def get_tissue_specific_events(alt_data_dict,
                               tissue_of_interest):
    """
    Giver alternative splicing_old data, tissue of interest, return the relevant data for the tissue of interest.
    :param alt_data_dict: Dict, dictionary with alternative splicing_old data returned from load_data_alt_const() method
    of this class.
    :param tissue_of_interest: str, which tissue we want to analyze.
    :return: Dict, a dictionary with following information about the tissue of interest:
    Key: tissue_of_interest + '_data' -> np.array, Feature values for events from the tissue of interest.
    Key: tissue_of_interest + '_identifier' -> np.array, event ids for events from the tissue of interest.
    Key: tissue_of_interest + '_labels' -> np.array, PSI1, PSI2, DPSI labels for events from the tissue of interest.
    Key: tissue_of_interest + '_tissue_1_set' -> np.array, one hot vector encoding for tissue 1 of events from the
                            tissue of interest.
    Key: tissue_of_interest + '_tissue_2_set' -> np.array, one hot vector encoding for tissue 2 of events from the
                            tissue of interest.
    Key: tissue_of_interest + '_exc_idx' -> np.array, Indices of differentially excluded events in the
                            tissue of interest.
    Key: tissue_of_interest + '_inc_idx' -> np.array, Indices of differentially included events in the
                            tissue of interest.
    Key: tissue_of_interest + 'nc_change' -> np.array, Indices of not changing events in the tissue of interest.
    Key: tissue_of_interest + '_feature_names' -> np.array, feature names for the selected features
    """
    # Get tissue pair ids, the order is defined by TISSUE_PAIRS variable
    tissue_pair_ids = alt_data_dict['labels_with_tissue_and_event_id'][:, 0]
    # Get labels (PSI1, PSI2, DeltaPSI (PSI1 - PSI2))
    labels = alt_data_dict['labels_with_tissue_and_event_id'][:, 2:5]
    # Get event ids for all the events
    event_ids = alt_data_dict['labels_with_tissue_and_event_id'][:, 1]
    # Use function convert the tissue_pair_ids to tissue pair names. eg. 0 -> Heart_Spleen,
    # the order is defined by TISSUE_PAIRS variable
    tissue_names = get_tissue_names(tissue_pair_ids)
    # Find idx where tissue of interest is the tissue 1
    relevant_idx = np.where(tissue_names[:, 0] == tissue_of_interest)[0]
    # Find idx where tissue of interest is the tissue 2 (has to be reversed before using)
    relevant_idx_reverse = np.where(tissue_names[:, 1] == tissue_of_interest)[0]
    # Reverse the labels for events where tissue of interest is the tissue 2
    corr_labels, corr_tissue_set_1, corr_tissue_set_2 = get_reverse_corrected_labels(labels,
                                                                                     alt_data_dict['tissue_1_set'],
                                                                                     alt_data_dict['tissue_2_set'],
                                                                                     relevant_idx_reverse)
    # Combine indices for all the events with tissue of interest (straight and reversed)
    all_relevant_idx = np.concatenate((relevant_idx, relevant_idx_reverse))
    # Store all the important data in this dict
    data_dict = dict()
    # features for the events from the tissue of interest
    data_dict[tissue_of_interest + '_data'] = alt_data_dict['data'][all_relevant_idx, :]
    # event ids for the events from the tissue of interest
    data_dict[tissue_of_interest + '_identifier'] = event_ids[all_relevant_idx]
    # labels for the events from the tissue of interest
    data_dict[tissue_of_interest + '_labels'] = corr_labels[all_relevant_idx, :]
    # tissue-1 one hot vector for the events from the tissue of interest
    data_dict[tissue_of_interest + '_tissue_1_set'] = corr_tissue_set_1[all_relevant_idx, :]
    # tissue-2 one hot vector for the events from the tissue of interest
    data_dict[tissue_of_interest + '_tissue_2_set'] = corr_tissue_set_2[all_relevant_idx, :]
    # differentially excluded event ids in the tissue of interest
    exc_idx = np.where(data_dict[tissue_of_interest + '_labels'][:, 2] >= 0.20)[0]
    # Differentially included event ids in the tissue of interest
    inc_idx = np.where(data_dict[tissue_of_interest + '_labels'][:, 2] <= -0.20)[0]
    # Not changing event ids in the tissue of interest
    nc_change = np.where(np.logical_and(data_dict[tissue_of_interest + '_labels'][:, 2] >= -0.05,
                                        data_dict[tissue_of_interest + '_labels'][:, 2] <= 0.05))[0]
    # Store differentially excluded, included and not changing idx
    data_dict[tissue_of_interest + '_exc_idx'] = exc_idx
    data_dict[tissue_of_interest + '_inc_idx'] = inc_idx
    data_dict[tissue_of_interest + 'nc_change'] = nc_change
    # Store selected feature names
    data_dict[tissue_of_interest + '_feature_names'] = alt_data_dict['feature_names']
    # Store overall psi class
    return data_dict


def get_constitutive_events(const_data_dict):
    """
    Given alternative versus constitutive data dict, return constitutive event ids and
    constitutive features
    :param const_data_dict: Dict, a dictionary with alternative versus constitutive data
    :return: np.array, event ids for constitutive events, np.array, features for the constitutive events
    """
    # All events with label 1 are constitutive events
    const_event_ids = np.where(const_data_dict['labels'][:, 0] >= 0.5)[0]
    # All features for constitutive events
    const_data_values_all = const_data_dict['data'][const_event_ids, :]
    # Get identifiers for constitutive events
    event_ids = const_data_dict['identifier'][const_event_ids]
    return event_ids, const_data_values_all


def get_inc_events(data_dict, tissue_of_interest):
    """
    Given the alternative splicing_old data_dict and the tissue of interest, return events differentially included in that
    tissue in comparison with other tissues
    :param data_dict: Dict, Alternative splicing_old data dict
    :param tissue_of_interest: str, which tissue we are looking at
    :return: np.array, features for events that have differential inclusion in the tissue of interest w.r.t some other
    tissue
    """
    # get differentially excluded event indices
    inc_events_idx = data_dict[tissue_of_interest + '_inc_idx']
    # get the differentially excluded features
    data = data_dict[tissue_of_interest + '_data']
    events_id = np.array(data_dict[tissue_of_interest + '_identifier'])
    inc_events = data[inc_events_idx, :]
    inc_ids = events_id[inc_events_idx]
    inc_events = {inc_ids[ii]: inc_events[ii] for ii in range(len(inc_ids))}
    inc_events = np.array([inc_events[ii] for ii in inc_events.keys()])
    return inc_events


def get_samples(tissue_data,
                tissue_of_interest,
                sample_type):
    """
    Get the differentially included events in the tissue of interest
    :param tissue_data: Dict, returned from get_tissue_specific_events() method in splicing_lig_baselines_attributions.py
    :param tissue_of_interest: str, tissue we are interested in
    :param sample_type: _inc_idx, _exc_idx, _nc_change
    :return: Dict, keys: event ids, values: features. Dict, keys: event ids, values: tissue-1 one hot vector encoding.
    Dict, keys: event ids, values, tissue-2 one hot vector encoding
    """
    # Get differentially included indices for the issue of interest
    inc_idx = tissue_data[tissue_of_interest + sample_type]
    if sample_type == SAMPLE_TYPE_NC:
        inc_idx = inc_idx[0:MAX_SAMPLES]
    # Get features for the differentially included events
    samples_inc = tissue_data[tissue_of_interest + "_data"][inc_idx, :]
    # Get event ids, tissue-1 one hot vector array and tissue-2 one hot vector array for the
    # differentially included events in  the tissue of interest
    inc_identifier = tissue_data[tissue_of_interest + '_identifier'][inc_idx]
    tissue_1_inc = tissue_data[tissue_of_interest + "_tissue_1_set"][inc_idx, :]
    tissue_2_inc = tissue_data[tissue_of_interest + "_tissue_2_set"][inc_idx, :]
    # Store features, tissue-1 one hot vector array and tissue-2 one hot vector array as dictionaries
    samples = []
    tissue_1 = []
    tissue_2 = []
    for i in range(len(inc_identifier)):
        samples.append(samples_inc[i])
        tissue_1.append(tissue_1_inc[i])
        tissue_2.append(tissue_2_inc[i])
    samples = np.array(samples)
    tissue_1 = np.array(tissue_1)
    tissue_2 = np.array(tissue_2)
    return samples, tissue_1, tissue_2, inc_identifier

