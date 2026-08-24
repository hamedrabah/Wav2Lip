"""Small ElevenLabs text-to-speech client used by Wav2Lip inference."""

from __future__ import print_function

import json
import os
from shutil import copyfileobj
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"


class ElevenLabsError(RuntimeError):
	"""Raised when ElevenLabs cannot generate the requested speech."""


def synthesize_speech(text, voice_id, api_key, output_path,
					  model_id="eleven_multilingual_v2",
					  output_format="mp3_44100_128",
					  api_url=ELEVENLABS_API_URL):
	"""Generate speech with ElevenLabs and save the returned audio.

	Args:
		text: Text that should be spoken.
		voice_id: ElevenLabs voice identifier.
		api_key: ElevenLabs API key. Callers should read it from the environment.
		output_path: File path for the generated audio.
		model_id: ElevenLabs text-to-speech model identifier.
		output_format: ElevenLabs output format, such as ``mp3_44100_128``.
		api_url: Override used by tests and compatible ElevenLabs endpoints.

	Returns:
		The output path.
	"""
	if not text or not text.strip():
		raise ValueError("ElevenLabs text cannot be empty")
	if not voice_id or not voice_id.strip():
		raise ValueError("ElevenLabs voice ID cannot be empty")
	if not api_key or not api_key.strip():
		raise ValueError("ElevenLabs API key cannot be empty")

	url = "{}/{}?{}".format(
		api_url.rstrip("/"),
		quote(voice_id.strip(), safe=""),
		urlencode({"output_format": output_format}),
	)
	payload = json.dumps({"text": text, "model_id": model_id}).encode("utf-8")
	request = Request(
		url,
		data=payload,
		headers={
			"Accept": "audio/mpeg",
			"Content-Type": "application/json",
			"xi-api-key": api_key,
		},
		method="POST",
	)

	try:
		with urlopen(request, timeout=120) as response:
			output_dir = os.path.dirname(os.path.abspath(output_path))
			if not os.path.isdir(output_dir):
				os.makedirs(output_dir)
			with open(output_path, "wb") as audio_file:
				copyfileobj(response, audio_file)
	except HTTPError as error:
		try:
			details = error.read().decode("utf-8", "replace")[:500]
		finally:
			error.close()
		raise ElevenLabsError(
			"ElevenLabs request failed with HTTP {}: {}".format(
				error.code, details or error.reason
			)
		)
	except URLError as error:
		raise ElevenLabsError(
			"Could not connect to ElevenLabs: {}".format(error.reason)
		)

	return output_path
