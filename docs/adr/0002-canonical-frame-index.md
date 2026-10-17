# One canonical frame index: 0-indexed frames of the standardized video

Every intermediate (cage video, detections, bird-crop video, predictions, ground truth) refers to frames by the **canonical frame**: the 0-indexed frame of the video after standardization to 15 fps. Media built by new code keeps every frame, writing a placeholder where a cage mask is empty, and media that can't (bird-crop videos contain only boxed frames) carries an explicit frame map back to canonical frames. Label Studio's 1-indexed frames are converted only where we read from or write to Label Studio, via `bird_cv.core.frames`.

## Consequences

The existing Label Studio pipelines (`bird_cv/pipelines/*.py` as of `pipeline-build`) keep their current frame behavior, including frame dropping in `crop_cages` and the indexing in `run_segment` and `predict_video_segments`: those are deliberate, tuned to how Label Studio displays and interpolates media. New code paths opt into canonical behavior through keyword-only arguments or separate helpers; they never change the defaults the existing pipelines rely on.

Converting existing Label Studio frames with a fixed −1 offset is only valid for media that is one-to-one with its standardized video: 15 fps and nothing dropped. Code that reads existing exports checks this per task (`bird_cv.core.frames.is_aligned_with_standard`) and excludes, with a count, any video that fails rather than silently misaligning it.
