from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from app.artifacts import ArtifactStore
from app.imports import ImportStore
from app.media_validation import validate_reference_media
from PIL import Image
import base64
import hashlib


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class ReferenceMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make(self, name, args):
        path = self.root / name
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args, str(path)], check=True)
        return path

    def test_video_rate_and_sound_are_reported(self):
        for fps in (24, 30, 60):
            with self.subTest(fps=fps):
                path = self.make(f"{fps}.mp4", ["-f", "lavfi", "-i", f"testsrc2=size=64x64:rate={fps}:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-c:a", "aac"])
                media = validate_reference_media(path, "video/mp4")
                self.assertEqual(media["source_fps"], fps)
                self.assertEqual(media["conditioning_fps"], 24)
                self.assertTrue(media["has_audio"])
                self.assertAlmostEqual(media["duration_seconds"], 2, delta=0.05)

    def test_audio_formats_and_invalid_duration(self):
        for filename, codec, media_type in (("a.wav", "pcm_s32le", "audio/wav"), ("a.mp3", "libmp3lame", "audio/mpeg")):
            path = self.make(filename, ["-f", "lavfi", "-i", "sine=duration=2", "-c:a", codec])
            self.assertTrue(validate_reference_media(path, media_type)["has_audio"])
        path = self.make("short.wav", ["-f", "lavfi", "-i", "sine=duration=1", "-c:a", "pcm_s16le"])
        with self.assertRaisesRegex(ValueError, "2_to_15"):
            validate_reference_media(path, "audio/wav")

    def test_real_png_chunk_commit_and_corrupt_bytes(self):
        path = self.make("input.png", ["-f", "lavfi", "-i", "color=size=64x64", "-frames:v", "1"])
        payload = path.read_bytes()
        store = ImportStore(ArtifactStore(self.root / "store"), validate_reference_media)
        sha = hashlib.sha256(payload).hexdigest()
        prepared = store.prepare(project_id="project-test", idempotency_key="png", filename="input.png",
            size=len(payload), sha256=sha, purpose="scene")
        args = {"project_id": "project-test", "import_id": prepared["import_id"]}
        store.chunk(**args, offset=0, content_base64=base64.b64encode(payload).decode(), sha256=sha)
        result = store.commit(**args)
        self.assertEqual(result["media"]["width"], 64)
        self.assertEqual(result["artifact"]["sha256"], sha)
        path.write_bytes(b"not a decodable image")
        with self.assertRaisesRegex(ValueError, "decode_failed"):
            validate_reference_media(path, "image/png")

    def test_media_type_cannot_disguise_video_as_image(self):
        path = self.make("video.mp4", ["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24:duration=2", "-c:v", "libx264"])
        with self.assertRaisesRegex(ValueError, "image_format_mismatch"):
            validate_reference_media(path, "image/png")

    def test_display_orientation_and_animated_images(self):
        image = Image.new("RGB", (64, 96), "red")
        exif = Image.Exif(); exif[274] = 6
        path = self.root / "oriented.jpg"
        image.save(path, exif=exif)
        media = validate_reference_media(path, "image/jpeg")
        self.assertEqual((media["width"], media["height"]), (96, 64))
        animated = self.root / "animated.png"
        image.save(animated, save_all=True, append_images=[Image.new("RGB", (64, 96), "blue")], duration=100)
        with self.assertRaisesRegex(ValueError, "still|format_mismatch"):
            validate_reference_media(animated, "image/png")
        source = self.make("plain.mp4", ["-f", "lavfi", "-i", "testsrc2=size=64x96:rate=24:duration=2", "-c:v", "libx264"])
        rotated = self.make("rotated.mp4", ["-display_rotation", "90", "-i", str(source), "-c", "copy"])
        media = validate_reference_media(rotated, "video/mp4")
        self.assertEqual((media["width"], media["height"]), (96, 64))


if __name__ == "__main__":
    unittest.main()
