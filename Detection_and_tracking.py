from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv

#model = YOLO('best.pt')
model = YOLO ('2000_img.pt')

highway = "testingFootage/prueba.mp4"
tCircle = "testingFootage/traffic_circle_footage.mp4"
#results = model.predict(source="1", show=True, conf = 0.45, classes = 0)
results = model.track(source = highway,show=True,conf = 0.25, tracker = "bytetrack.yaml")
#print(results)

class VideoProcessor:
    def __init__(
        self,
        roboflow_api_key: str,
        model_id: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = results
        #self.tracker = sv.ByteTrack()



    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
