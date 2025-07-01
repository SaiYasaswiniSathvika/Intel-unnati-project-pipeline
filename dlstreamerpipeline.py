import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import signal

# Initialize GStreamer
Gst.init(None)

class DLStreamerPipeline:
    def __init__(self, video_path):
        self.pipeline = None
        self.loop = GLib.MainLoop()
        self.video_path = video_path

        # Define the DL Streamer GStreamer pipeline
        pipeline_str = f"""
        filesrc location={self.video_path} !
        decodebin !
        videoconvert !
        gvadetect model=intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml device=CPU !
        gvafpscounter !
        fakesink
        """

        self.pipeline = Gst.parse_launch(pipeline_str.strip())

    def run(self):
        # Handle interrupt
        signal.signal(signal.SIGINT, self.stop)

        # Set pipeline state to PLAYING
        self.pipeline.set_state(Gst.State.PLAYING)
        print("🔁 DL Streamer pipeline started. Press Ctrl+C to stop.")
        try:
            self.loop.run()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.stop()

    def stop(self, *args):
        print("🛑 Stopping pipeline...")
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()

if __name__ == "__main__":
    pipeline = DLStreamerPipeline("pedestrian.avi")
    pipeline.run()
