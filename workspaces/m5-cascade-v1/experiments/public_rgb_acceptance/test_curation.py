from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from build_media import safe_member, verify_frame_numbers, jpeg_information, inspect_video
from common import DATA, ROOT, REPORT, LABELS, SHA, ANN_SHA, ChunksReader, read, sha
from prepare_plan import source_annotations


class PublicRGBCurationTests(unittest.TestCase):
    def test_frozen_plan_coverage_and_honest_head_involvement(self):
        plan=read(ROOT/"frozen-record-plan.json")
        self.assertEqual(sha(ROOT/"frozen-record-plan.json"),read(REPORT/"record-plan-fingerprint.json")["sha256"])
        self.assertEqual(len(plan["records"]),26)
        self.assertEqual(len({r["recording_group"] for r in plan["records"]}),26)
        self.assertEqual({i for r in plan["records"] for i in r["original_labels"]},set(LABELS))
        self.assertEqual(plan["current_head_split_counts"],{"train":20,"validation":2,"test":4})
        self.assertTrue(all(not r["independent_unseen_test_claim"] for r in plan["records"]))
        self.assertTrue(all(r["current_use"]=="engineering_regression_only" for r in plan["records"] if r["used_for_current_head_training"] or r["used_for_current_head_tuning"]))
        self.assertEqual(sha(DATA/"improvement-v2/combined-64.pt"),plan["current_head_checkpoint_sha256"])
        self.assertEqual(sha(DATA/"improvement-v2/record-splits.json"),plan["current_head_split_sha256"])

    def test_original_annotation_agrees_with_every_planned_record(self):
        plan=read(ROOT/"frozen-record-plan.json")
        self.assertEqual(sha(DATA/"assets/ann.tar.gz"),ANN_SHA)
        annotations=source_annotations(DATA/"assets/ann.tar.gz",{r["annotation_frame_dir"] for r in plan["records"]})
        for record in plan["records"]:
            actual=annotations[record["annotation_frame_dir"]]
            self.assertEqual(actual["labels"],record["original_labels"])
            self.assertEqual(actual["frame_count"],record["expected_frames"])

    def test_only_declared_single_view_jpegs_can_be_selected(self):
        selected={"SupC14_Dataset/SupC00/r00/ch0":{"record_id":"SupC00/r00/ch0"}}
        result=safe_member("./SupC14_Dataset/SupC00/r00/ch0/00000001.jpg",100,selected)
        self.assertEqual(result[1],"00000001.jpg")
        self.assertIsNone(safe_member("./SupC14_Dataset/SupC00/r00/ch1/00000001.jpg",100,selected))
        self.assertIsNone(safe_member("./SupC14_Dataset/SupC00/r00/ofch0/flow_x_00001.jpg",100,selected))
        for name in ("../SupC14_Dataset/SupC00/r00/ch0/00000001.jpg","/SupC14_Dataset/SupC00/r00/ch0/00000001.jpg"):
            with self.assertRaisesRegex(ValueError,"Unsafe"):
                safe_member(name,100,selected)
        with self.assertRaises(ValueError):
            safe_member("SupC14_Dataset/SupC00/r00/ch0/payload.py",100,selected)
        with self.assertRaises(ValueError):
            safe_member("SupC14_Dataset/SupC00/r00/ch0/00000001.jpg",5*2**20,selected)

    def test_missing_duplicate_and_truncated_frames_are_rejected(self):
        def images(numbers):return [{"filename":f"{n:08d}.jpg"} for n in numbers]
        self.assertEqual(verify_frame_numbers(images([3,1,2]),3),1)
        for numbers in ([1,2],[1,2,2],[1,2,4]):
            with self.assertRaises(ValueError):verify_frame_numbers(images(numbers),3)
        output=BytesIO();Image.new("RGB",(32,24),(12,34,56)).save(output,format="JPEG")
        data=output.getvalue();self.assertEqual(jpeg_information(data),(24,32))
        with self.assertRaises(ValueError):jpeg_information(data[:-20])

    def test_chunk_reader_preserves_bytes_across_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/str(i) for i in range(3)]
            for path,data in zip(paths,[b"abc",b"defg",b"h"]):path.write_bytes(data)
            reader=ChunksReader(paths)
            self.assertEqual(reader.read(2),b"ab")
            self.assertEqual(reader.read(5),b"cdefg")
            self.assertEqual(reader.read(5),b"h")
            self.assertEqual(reader.read(5),b"")
            with self.assertRaises(ValueError):reader.read()
            reader.close()

    def test_video_full_decode_checks_frame_count_and_timebase(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"images").mkdir()
            video=root/"video.mp4"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),25.,(32,24))
            self.assertTrue(writer.isOpened())
            images=[]
            for index in range(8):
                frame=np.full((24,32,3),40+index,dtype=np.uint8)
                filename=f"{index+1:08d}.jpg";cv2.imwrite(str(root/"images"/filename),frame)
                images.append({"filename":filename});writer.write(frame)
            writer.release()
            manifest={"expected_frames":8,"image_shape":[24,32],"images":images}
            self.assertEqual(inspect_video(video,manifest)["decoded_frames"],8)
            with self.assertRaisesRegex(ValueError,"frame count/timebase"):
                inspect_video(video,{**manifest,"expected_frames":9})
            with self.assertRaisesRegex(ValueError,"dimensions"):
                inspect_video(video,{**manifest,"image_shape":[16,32]})
            bad_fps=root/"bad-fps.mp4"
            writer=cv2.VideoWriter(str(bad_fps),cv2.VideoWriter_fourcc(*"mp4v"),20.,(32,24))
            self.assertTrue(writer.isOpened())
            for _ in range(8):writer.write(np.full((24,32,3),40,dtype=np.uint8))
            writer.release()
            with self.assertRaisesRegex(ValueError,"frame count/timebase"):
                inspect_video(bad_fps,manifest)

    def test_frozen_collection_cannot_be_silently_overwritten(self):
        import build_media
        import finalize_collection
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"collection-manifest.json").write_text("{}")
            with patch.object(build_media,"ROOT",root),patch("sys.argv",["build_media.py","videos"]):
                with self.assertRaisesRegex(FileExistsError,"frozen"):
                    build_media.main()
            with patch.object(finalize_collection,"ROOT",root):
                with self.assertRaisesRegex(FileExistsError,"frozen"):
                    finalize_collection.main()


if __name__=="__main__":unittest.main(verbosity=2)
