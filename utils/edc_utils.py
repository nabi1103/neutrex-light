"""Contains the "compute_edc" function that computes the data for an EDC plot."""

from typing import Optional

import numpy as np
from tqdm import tqdm
np.random.seed(1)

#import registry

_dsdc_error_modes = {"Mean-DSDC"}


def is_dsdc_error_mode(error_mode: str):
  return error_mode in _dsdc_error_modes



def compute_edc(
    error_mode: str,
    threshold_quantile: Optional[float] = None,
    database_quality_id: Optional[str] = None,
    database_frscore_id: Optional[str] = None,
    quality_scores: Optional[np.ndarray] = None,
    comparison_scores: Optional[np.ndarray] = None,
    discover_file_path: str = "",
    verbose: bool = False,
    pair_quality_mode: str = "min",
    oracle_scaling: Optional[dict] = None,
) -> dict:
  """Computes EDC plot data (but doesn't create the plot itself).

  Parameters
  ----------
  database_quality_id : Optional[str]
    The ID of the Database-quality instance that is to be used for the EDC computation.
    Alternatively to database_quality_id & database_frscore_id, you can also pass comparison_scores & quality_scores.
  database_frscore_id : Optional[str]
    The ID of the Database-frscore instance that is to be used for the EDC computation.
    Alternatively to database_quality_id & database_frscore_id, you can also pass comparison_scores & quality_scores.
  error_mode : str
    Either "FNMR", "FMR", or "Mean-DSDC".
    The relevant pairs will be selected from the Database-frscore, i.e. mated pairs for FNMR, non-mated pairs for FMR.
    "Mean-DSDC" is a special mode, which may or may not be considered as an "EDC", see the _edc_core documentation.
  threshold_quantile : Optional[float]
    The [0,1] quantile used to obtain the FR comparison score threshold for the EDC computation.
    This is equivalent to the starting FNMR/FMR error value, i.e. the error value when no pairs are filtered by quality.
    This value is required for regular EDCs, but not allowed for DSDCs.
  discover_file_path : str
    If not "", discover_and_register_files will be called using this path for Database-quality/frscore.
  verbose : bool
    If True, print a small amount of general information (image pair count & FR comparison threshold).
    Warnings on incorrect arguments are always printed.
  pair_quality_mode : str
    Either "min", "max", or "mean".
    Specifies the function used to obtain the pairwise quality score for each ImagePair in the Database-frscore.
  oracle_scaling : Optional[dict]
    This parameter can take EDC oracle dictionary output from `compute_oracle_edc`.
    If passed, the EDC "error" results will be scaled relative to the best-case oracle error results
    and the (constant) initial (0%-discard) error value.
    These scaled results are stored in a sub-dicitonary with key "oracle_scaled" as "error"/"discarded",
    analogous to (and in addition to) the usual "error"/"discarded" results.

    The corresponding oracle errors are subtracted from the FIQAA errors,
    then the result is divided by the difference between the oracle errors
    and the (constant) initial (0%-discard) error value.
    Results are only stored for points where the oracle error is below the initial error value
    (which should ususally be the case, but is not always guranteed, e.g. close to the 0%-discard point).

    Thus the scaled results will usually be in the [0, 1] range, but they may fall outside that range too
    (either by being better than the oracle, i.e. below 0, or by being worse than the initial error, i.e. above 1).

    The constant initial error approximately represents the (average of) randomly assigning quality scores to images
    (which can be demonstrated via `compute_random_edc`), so it is used as the realistic worst-case.
    (The actual worst-case could be approximated via the old `compute_oracle_edc` function too.
    These worst-case oracle values could grow substantially higher than the initial error for higher discard rates.
    But using the actual worst-case approximation makes less practical sense for the scaling part here,
    since a real FIQA shouldn't be worse than a random FIQAA in the first place, and thus the approximation of the
    average of infinite random FIQAAs should be used as the realistic ceiling/"worst-case".)
  comparison_scores : Optional[np.ndarray]
    Instead of database_frscore_id/database_quality_id, you can pass comparison_scores/quality_scores directly.
    In this case the caller must ensure that the order of the two array does match, i.e. so that
    each comparison_scores/quality_scores entry corresponds to the same comparison pair.
    Note: Similarity scores are expected.
    Note: Passing comparison_scores/quality_scores arrays will disable the caching system.
    Note: Passing a mixture, i.e. database_quality_id + comparison_scores or database_frscore_id + quality_scores,
    will raise an `Exception`.
  quality_scores : Optional[np.ndarray]
    Pairwise quality_scores, see the comparison_scores parameter.

  Returns
  -------
  dict
    Raises an `Exception` if error_mode/pair_quality_mode/threshold_quantile is invalid (see `_check_arguments`).
    Returns a result dictionary otherwise:
      "config": Contains the given configuration in form of a dictionary.
      "error": The list of error (e.g. FNMR) values.
      "discarded": The list of discard fraction values.
      "oracle_scaled": Optional dictionary with oracle-scaled "error"/"discarded" lists, see "oracle_scaling".
      "stddev": Additional standard deviation information, only for "Mean-DSDC".
    The "error" and "discarded" lists correspond to each other and can be used directly to create the actual plot.
  """
  database_mode = database_quality_id is not None or database_frscore_id is not None
  if database_mode and (quality_scores is not None or comparison_scores is not None):
    raise Exception("Only database_quality_id+database_frscore_id or quality_scores+comparison_scores may be passed.")

  pair_quality_func = _check_arguments(error_mode, threshold_quantile, pair_quality_mode)

  results_config = {
      "threshold_quantile": threshold_quantile,
      "error_mode": error_mode,
      "pair_quality_mode": pair_quality_mode,
  }
  if database_mode:
    results_config["database_quality"] = database_quality_id
    results_config["database_frscore"] = database_frscore_id

  if discover_file_path != "":
    registry.discover_and_register_files({"Database-quality", "Database-frscore"}, discover_file_path)

  if database_mode:
    database_quality = registry.get_registry("Database-quality", database_quality_id)
    database_frscore = registry.get_registry("Database-frscore", database_frscore_id)

    # Select relevant image pairs
    image_pairs = _select_relevant_image_pairs(database_frscore, error_mode)

    # Assign per-pair quality & FR scores
    quality_dict = {image.dlid: image for image in database_quality.images}
    fr_quality_scores = np.zeros((2, len(image_pairs)))
    fr_comparison_scores = fr_quality_scores[0]
    quality_scores = fr_quality_scores[1]
    for i, image_pair in enumerate(image_pairs):
      key = image_pair.dlid_tuple
      image_quality0 = quality_dict[key[0]]
      quality0 = image_quality0.quality.scalar_quality_score
      image_quality1 = quality_dict[key[1]]
      quality1 = image_quality1.quality.scalar_quality_score
      fr_comparison_scores[i] = image_pair.frscore.cosine
      quality_scores[i] = pair_quality_func((quality0, quality1))
  else:
    fr_comparison_scores = comparison_scores
    # quality_scores is already assigned via the parameter

  # Compute the EDC values
  order_scores = np.argsort(quality_scores)
  quality_scores = quality_scores[order_scores]
  fr_comparison_scores = fr_comparison_scores[order_scores]
  if threshold_quantile is None:
    fr_comparison_threshold = None
  else:
    fr_comparison_threshold = np.quantile(fr_comparison_scores, threshold_quantile)

  results = _edc_core(
      error_mode,
      quality_scores,
      fr_comparison_scores,
      fr_comparison_threshold,
      verbose=verbose,
      oracle_scaling=oracle_scaling)
  results["config"] = results_config

  return results


def _check_arguments(error_mode: str, threshold_quantile: float, pair_quality_mode: str):
  """Checks whether the error_mode/threshold_quantile/pair_quality_mode arguments are valid.
  If they are, the function corresponding to the pair_quality_mode is returned.
  If not, an Exception is raised.
  """
  pair_quality_funcs = {"min": min, "max": max, "mean": np.mean}
  if error_mode not in {"FNMR", "FMR", *_dsdc_error_modes}:
    raise Exception(f"Invalid error_mode '{error_mode}', must be either FNMR or FMR.")
  if pair_quality_mode not in pair_quality_funcs:
    raise Exception(
        f"Invalid pair_quality_mode '{pair_quality_mode}', must be one of {list(pair_quality_funcs.keys())}.")
  if is_dsdc_error_mode(error_mode):
    if threshold_quantile is not None:
      raise Exception("The DSDC does not use any comparison score threshold, but a quantile for one was specified.")
  elif threshold_quantile < 0 or threshold_quantile > 1:
    raise Exception(f"Invalid threshold_quantile '{threshold_quantile}', must be in [0,1].")
  return pair_quality_funcs[pair_quality_mode]


def _select_relevant_image_pairs(database_frscore, error_mode: str) -> list:
  """Returns a list of the relevant (depending on the error_mode) image pairs within a Database-frscore."""
  if is_dsdc_error_mode(error_mode):
    image_pairs = list(database_frscore.image_pairs)
  else:
    same_subject_mode = error_mode == "FNMR"
    image_pairs = []
    for image_pair in database_frscore.image_pairs:
      mated = image_pair.image0.subject_dlid == image_pair.image1.subject_dlid
      if mated == same_subject_mode:
        image_pairs.append(image_pair)
  return image_pairs


def _get_fr_comparison_scores(image_pairs: list) -> np.ndarray:
  fr_comparison_scores = np.zeros(len(image_pairs))
  for i, image_pair in enumerate(tqdm(image_pairs, desc="Get FR comparison scores")):
    fr_comparison_scores[i] = image_pair.frscore.cosine
  return fr_comparison_scores


def _get_fr_comparison_threshold(image_pairs: list, threshold_quantile: float) -> tuple:
  fr_comparison_scores = _get_fr_comparison_scores(image_pairs)
  fr_comparison_threshold = np.quantile(fr_comparison_scores, threshold_quantile)
  return fr_comparison_threshold, fr_comparison_scores


def _create_image_pair_index_mappings(image_pairs: list) -> tuple:
  image_index_maximum = 0
  for image_pair in image_pairs:
    image_index_maximum = max(image_index_maximum, image_pair.dlid_tuple[0], image_pair.dlid_tuple[1])
  index_map = np.full(image_index_maximum + 1, -1, dtype=np.int32)
  max_map_index = 0
  for image_pair in tqdm(image_pairs, desc="Map pairwise image indices"):
    for i in image_pair.dlid_tuple:
      if index_map[i] < 0:
        index_map[i] = max_map_index
        max_map_index += 1
  return index_map, max_map_index


def _derive_pairwise_quality_from_array(pair_quality_func, image_pairs: list, index_map: np.ndarray,
                                        image_quality_scores: np.ndarray) -> np.ndarray:
  pairwise_quality_scores = np.zeros(len(image_pairs), dtype=image_quality_scores.dtype)
  for i, image_pair in enumerate(tqdm(image_pairs, desc="Derive pairwise quality")):
    mapped_index_0 = index_map[image_pair.dlid_tuple[0]]
    mapped_index_1 = index_map[image_pair.dlid_tuple[1]]
    quality0, quality1 = image_quality_scores[mapped_index_0], image_quality_scores[mapped_index_1]
    pairwise_quality_scores[i] = pair_quality_func((quality0, quality1))
  return pairwise_quality_scores


def _sort_scores(pairwise_quality_scores, fr_comparison_scores):
  order_scores = np.argsort(pairwise_quality_scores)
  pairwise_quality_scores = pairwise_quality_scores[order_scores]
  fr_comparison_scores = fr_comparison_scores[order_scores]
  return pairwise_quality_scores, fr_comparison_scores


def _form_error_comparison_decision(error_mode: str,
                                    fr_comparison_score_or_scores,
                                    fr_comparison_threshold: float,
                                    out: Optional[np.ndarray] = None):
  if error_mode == "FNMR":
    # FNMR, so non-matches are errors
    return np.less(fr_comparison_score_or_scores, fr_comparison_threshold, out=out)
  elif error_mode == "FMR":
    # FMR, so matches are errors
    return np.greater_equal(fr_comparison_score_or_scores, fr_comparison_threshold, out=out)


_edc_core__verbose_cache = None  # This is used to only print new "verbose" _edc_core info if it actually changed.


def clear_verbose_cache():
  global _edc_core__verbose_cache  # pylint: disable=global-statement
  _edc_core__verbose_cache = None


def _dsdc_compute_running_stats(pair_comparison_scores):
  comparison_count = len(pair_comparison_scores)
  mean_results = np.zeros(comparison_count, dtype=np.float64)
  stddev_results = np.zeros(comparison_count, dtype=np.float64)
  mean_accumulator = 0
  stddev_accumulator = 0
  for i, value in enumerate(pair_comparison_scores):
    j = i + 1
    temp_mean = mean_accumulator
    mean_accumulator += (value - temp_mean) / j
    stddev_accumulator += (value - temp_mean) * (value - mean_accumulator)
    mean_results[i] = mean_accumulator
    if j > 1:
      stddev_results[i] = np.sqrt(stddev_accumulator / (j - 1))  # Would be j instead of j-1 for whole population.
    else:
      stddev_results[i] = 0
  return mean_results, stddev_results


def _edc_core(
    error_mode: str,
    pair_quality_scores: np.ndarray,
    pair_comparison_scores: np.ndarray,
    fr_comparison_threshold: Optional[float],
    verbose: bool = False,
    with_error_per_discard_count: bool = False,
    oracle_scaling: Optional[dict] = None,
    tqdm_desc: str = "EDC",
):
  """This contains the actual EDC computation.
  The pair_comparison_scores are linked to the pair_quality_scores, and the scores must be sorted by the quality scores.

  This also supports the computation of a "Mean-DSDC", "Mean-Dissimilarity-Score-versus-Discard-Characteristic",
  which is a (presumably) new metric prototype added to this framework.
  The DSDC is basically the EDC with some comparison score statistic (e.g. the mean) as the "error"
  (the "dissimilarity score" is used to make sure that lower values are better,
  but since this function expects similarity scores as input these have to be inverted).
  The point of this is to make the evaluation independent of the fr_comparison_threshold parameter,
  which is not used at all for the DSDC (an Exception will be raised if fr_comparison_threshold is not None).
  """
  if with_error_per_discard_count and oracle_scaling is not None:
    raise Exception("with_error_per_discard_count requires disabled oracle_scaling")

  assert len(pair_quality_scores) == len(pair_comparison_scores), "Input quality/comparison score count mismatch"

  dsdc_mode = is_dsdc_error_mode(error_mode)  # See "DSDC" in the function description.

  if dsdc_mode:
    if tqdm_desc == "EDC":
      tqdm_desc = "DSDC"
    if fr_comparison_threshold is not None:
      raise Exception("The DSDC does not use any comparison score threshold, but one was specified.")

  comparison_count = len(pair_quality_scores)
  if verbose:
    global _edc_core__verbose_cache  # pylint: disable=global-statement
    verbose_value = (comparison_count, fr_comparison_threshold)
    if _edc_core__verbose_cache != verbose_value:
      _edc_core__verbose_cache = verbose_value
      print(f"Image pairs: {comparison_count}")
      if not dsdc_mode:
        print(f"FR threshold: {fr_comparison_threshold}")

  if verbose:
    pbar = tqdm(total=4, desc=tqdm_desc)

  if dsdc_mode:
    # Normalize the comparison scores:
    pair_comparison_scores__min = min(pair_comparison_scores)
    pair_comparison_scores__max = max(pair_comparison_scores)
    pair_comparison_scores__range = pair_comparison_scores__max - pair_comparison_scores__min
    pair_comparison_scores = (pair_comparison_scores - pair_comparison_scores__min) / pair_comparison_scores__range

    # The input pair_comparison_scores are expected to be similarity scores, not dissimilarity scores.
    # Invert the comparison scores so that lower values will be better (as they should be for an EDC):
    pair_comparison_scores = 1 - pair_comparison_scores

    # Compute comparison score statistics:
    error_counts, stddev_results = _dsdc_compute_running_stats(np.flipud(pair_comparison_scores))
    error_counts = np.flipud(error_counts)  # NOTE These are actually the comparison score mean values for this mode.
    stddev_results = np.flipud(stddev_results)
  else:
    error_counts = np.zeros(comparison_count, dtype=np.uint32)
    _form_error_comparison_decision(error_mode, pair_comparison_scores, fr_comparison_threshold, out=error_counts)
    error_counts = np.flipud(np.cumsum(np.flipud(error_counts), out=error_counts))
  if verbose:
    pbar.update(1)

  discard_counts = np.where(pair_quality_scores[:-1] != pair_quality_scores[1:])[0] + 1
  discard_counts = np.concatenate(([0], discard_counts))
  if verbose:
    pbar.update(1)

  if dsdc_mode:
    # NOTE Technically there are no "error" counts & fractions for this mode.
    error_fractions = error_counts[discard_counts]
  else:
    remaining_counts = comparison_count - discard_counts
    error_fractions = error_counts[discard_counts] / remaining_counts
  discard_fractions = discard_counts / comparison_count
  if verbose:
    pbar.update(1)

  results = {
      "error": error_fractions,
      "discarded": discard_fractions,
      "starting_error_count": error_counts[0],
      "comparison_count": comparison_count,
  }
  if dsdc_mode:
    results["stddev"] = stddev_results[discard_counts]
  if oracle_scaling is not None:
    starting_error_constant = error_counts[0] / comparison_count
    oracle_best = oracle_scaling["error_per_discard_count"]
    o__discard_counts = discard_counts[oracle_best[discard_counts] < starting_error_constant]
    o__remaining_counts = comparison_count - o__discard_counts
    o__error_fractions = error_counts[o__discard_counts] / o__remaining_counts
    o__oracle_best = oracle_best[o__discard_counts]
    o__error_fractions -= o__oracle_best
    o__error_fractions /= starting_error_constant - o__oracle_best
    o__discard_fractions = o__discard_counts / comparison_count
    results["oracle_scaled"] = {
        "error": o__error_fractions,
        "discarded": o__discard_fractions,
    }
  elif with_error_per_discard_count:
    # Compute error_per_discard_count (with stepwise interpoaltion):
    error_per_discard_count = np.full(comparison_count, -1, dtype=np.float64)
    index = 0
    last_error_fraction = error_fractions[0]
    for next_index, next_error_fraction in zip(discard_counts, error_fractions):
      while index < next_index:
        error_per_discard_count[index] = last_error_fraction
        index += 1
      last_error_fraction = next_error_fraction
    while index < comparison_count:
      error_per_discard_count[index] = last_error_fraction
      index += 1
    results["error_per_discard_count"] = error_per_discard_count
  if verbose:
    pbar.update(1)
    pbar.close()

  return results


def compute_edc_pauc(edc_results: dict, discard_fraction_limit: float = 1):
  """The computes the (rectangular/stepped) AUC or pAUC for the given edc_results & discard_fraction_limit.

  Note: This does not subtract the "(area under) theoretical best" (max(0, error at 0%-discard minus discard fraction)),
  as done in the paper "Finger image quality assessment features - definitions and evaluation".
  Instead the (p)AUC can be computed on the oracle-scaled edc_results, which is presumably more usefuly for analysis,
  since the best-case oracle should be a better approximation of an optimal FIQAA than this "theoretical best" line.

  Parameters
  ----------
  edc_results : dict
    The EDC data as returned by the compute_edc function.
    Only the "error" and "discarded" fraction value arrays are used.
  discard_fraction_limit : float
    Must be in [0,1].
    If 1 (the default), the full AUC is computed.
    Otherwise the pAUC for that discard fraction limit will be computed.

  Returns
  -------
  float
    The computed (p)AUC.
  """
  error_fractions, discard_fractions = edc_results["error"], edc_results["discarded"]
  assert len(error_fractions) == len(discard_fractions), "error_fractions/discard_fractions length mismatch"
  assert discard_fraction_limit >= 0 and discard_fraction_limit <= 1, "Invalid discard_fraction_limit"
  if discard_fraction_limit == 0:
    return 0
  pauc = 0
  for i in range(len(discard_fractions)):  # pylint: disable=consider-using-enumerate
    if i == (len(discard_fractions) - 1) or discard_fractions[i + 1] >= discard_fraction_limit:
      pauc += error_fractions[i] * (discard_fraction_limit - discard_fractions[i])
      break
    else:
      pauc += error_fractions[i] * (discard_fractions[i + 1] - discard_fractions[i])
  return pauc
