import os
import sys
import json
import argparse

# Ensure we can import pipeline module when run from backend cwd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_video_pipeline import process_video


def main():
    parser = argparse.ArgumentParser(description="Process a video segment through the pipeline and report output paths.")
    parser.add_argument("--segment", required=True, help="Path to the video segment (e.g., sec_1.mp4)")
    parser.add_argument("--session", required=False, help="Optional session id")
    args = parser.parse_args()

    segment_path = args.segment
    session_id = args.session or "default"

    try:
        if not os.path.exists(segment_path):
            raise FileNotFoundError(f"Segment not found: {segment_path}")

        ok = process_video(segment_path)
        video_name = os.path.splitext(os.path.basename(segment_path))[0]
        aod_rel = os.path.join("outputs_vid", "aod", "vid", f"{video_name}.mp4")
        aod_abs = os.path.abspath(aod_rel)
        aod_url = f"/outputs/aod/vid/{video_name}.mp4"

        result = {
            "status": "ok" if ok else "failed",
            "sessionId": session_id,
            "videoName": video_name,
            "aodVideoPath": aod_abs,
            "aodVideoUrl": aod_url,
            "segmentIndex": int(''.join(filter(str.isdigit, video_name)) or '0')
        }
        print("AOD_RESULT:" + json.dumps(result), flush=True)
        sys.exit(0 if ok else 1)
    except Exception as e:
        err = {
            "status": "error",
            "message": str(e),
            "sessionId": session_id
        }
        print("AOD_RESULT:" + json.dumps(err), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
