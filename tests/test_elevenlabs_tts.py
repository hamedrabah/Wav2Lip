import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from elevenlabs_tts import ElevenLabsError, synthesize_speech


class FakeResponse(io.BytesIO):
	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.close()


class ElevenLabsTTSTest(unittest.TestCase):
	def test_sends_request_and_writes_audio(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			output_path = os.path.join(temp_dir, "speech.mp3")
			with patch("elevenlabs_tts.urlopen", return_value=FakeResponse(b"mp3-data")) as mocked_urlopen:
				result = synthesize_speech(
					"Hello from Wav2Lip",
					"voice/id",
					"secret-key",
					output_path,
				)

			self.assertEqual(result, output_path)
			with open(output_path, "rb") as audio_file:
				self.assertEqual(audio_file.read(), b"mp3-data")

			request = mocked_urlopen.call_args[0][0]
			self.assertEqual(mocked_urlopen.call_args[1]["timeout"], 120)
			self.assertIn("/voice%2Fid?output_format=mp3_44100_128", request.full_url)
			self.assertEqual(request.get_method(), "POST")
			self.assertEqual(request.headers["Xi-api-key"], "secret-key")
			self.assertEqual(
				json.loads(request.data.decode("utf-8")),
				{
					"text": "Hello from Wav2Lip",
					"model_id": "eleven_multilingual_v2",
				},
			)

	def test_rejects_missing_required_values(self):
		with self.assertRaises(ValueError):
			synthesize_speech("", "voice", "key", "speech.mp3")
		with self.assertRaises(ValueError):
			synthesize_speech("hello", "", "key", "speech.mp3")
		with self.assertRaises(ValueError):
			synthesize_speech("hello", "voice", "", "speech.mp3")

	def test_reports_api_error_without_exposing_key(self):
		error = HTTPError(
			"https://api.elevenlabs.io/v1/text-to-speech/voice",
			401,
			"Unauthorized",
			{},
			io.BytesIO(b'{"detail":"invalid key"}'),
		)
		with patch("elevenlabs_tts.urlopen", side_effect=error):
			with self.assertRaises(ElevenLabsError) as raised:
				synthesize_speech("hello", "voice", "secret-key", "speech.mp3")

		self.assertIn("HTTP 401", str(raised.exception))
		self.assertNotIn("secret-key", str(raised.exception))


if __name__ == "__main__":
	unittest.main()
