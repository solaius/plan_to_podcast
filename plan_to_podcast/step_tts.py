import re
import torch
import numpy as np
import torchaudio
from transformers import AutoProcessor, AutoModel
from typing import Optional, Tuple

class StepTTS:
    def __init__(self):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            print("Loading TTS processor...")
            self.processor = AutoProcessor.from_pretrained(
                "stepfun-ai/Step-Audio-TTS-3B",
                trust_remote_code=True,
                cache_dir="/workspace/models"
            )
            
            print("Loading TTS model...")
            self.model = AutoModel.from_pretrained(
                "stepfun-ai/Step-Audio-TTS-3B",
                trust_remote_code=True,
                cache_dir="/workspace/models"
            ).to(self.device)
            
            self.model.eval()
            print("TTS model loaded successfully!")
            
        except Exception as e:
            print(f"Error initializing TTS model: {str(e)}")
            raise

    def _load_reference_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess reference audio for voice cloning."""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to model's sample rate if needed
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                waveform = resampler(waveform)
            
            # Extract voice embedding
            with torch.no_grad():
                voice_embedding = self.model.extract_voice_embedding(waveform.to(self.device))
            
            return voice_embedding
            
        except Exception as e:
            print(f"Error loading reference audio: {str(e)}")
            return None

    def tts(self, text: str, reference_audio: Optional[str] = None, speed: float = 1.0):
        """Convert text to speech using Step-Audio-TTS-3B with optional voice cloning."""
        try:
            # Preprocess text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            # Load reference audio if provided
            voice_embedding = None
            if reference_audio:
                voice_embedding = self._load_reference_audio(reference_audio)
            
            # Generate audio
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Add voice embedding if available
            if voice_embedding is not None:
                inputs["voice_embedding"] = voice_embedding
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,
                    max_length=1000
                )
            
            # Get audio data from model output
            audio = outputs.audio[0].cpu().numpy()
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                # Simple resampling for speed adjustment
                target_length = int(len(audio) / speed)
                audio = np.interp(
                    np.linspace(0, len(audio), target_length),
                    np.arange(len(audio)),
                    audio
                )
            
            return (24000, audio)  # Return sample rate and audio data
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            # Return 1 second of silence as fallback
            return (24000, np.zeros(24000))

from plan_to_podcast.voice_manager import VoiceManager

def podcast_tts(text: str, host_voices: dict[str, str]):
    """Convert podcast script text to speech."""
    try:
        # Initialize TTS engine and voice manager
        print("Initializing TTS engine...")
        tts_engine = StepTTS()
        voice_manager = VoiceManager()
        
        # Parse script
        print("Parsing podcast script...")
        pattern = r"<\|(.*?)\|>: (.*?)(?:\n\n|$)"
        turns = re.findall(pattern=pattern, string=text.strip())
        
        if not turns:
            raise ValueError("No valid dialogue turns found in the script")
        
        # Validate speakers
        if not all(speaker in host_voices for speaker, _ in turns):
            non_matching_hosts = [speaker for speaker, _ in turns if speaker not in host_voices]
            raise ValueError(f"Invalid speaker(s): {set(non_matching_hosts)}")
        
        # Process each turn
        print(f"Processing {len(turns)} dialogue turns...")
        audio_segments = []
        tokens = []
        
        for i, (speaker, content) in enumerate(turns, 1):
            print(f"Processing turn {i}/{len(turns)} for {speaker}...")
            
            # Get voice reference audio if available
            voice_id = host_voices[speaker]
            reference_audio = voice_manager.get_voice_path(voice_id)
            
            # Generate speech
            turn_audio = tts_engine.tts(
                text=content,
                reference_audio=reference_audio,
                speed=1.0
            )
            audio_segments.append(turn_audio[1])
            tokens.append(f"{content}\n{speaker}")
            
            # Add a short pause between turns
            pause_length = int(24000 * 0.5)  # 0.5 second pause
            audio_segments.append(np.zeros(pause_length))
        
        # Concatenate all audio segments
        print("Combining audio segments...")
        audio = np.concatenate(audio_segments)
        audio_tokens = "\n\n".join(tokens)
        
        print("Audio generation complete!")
        return (24000, audio), audio_tokens
        
    except Exception as e:
        error_msg = f"Error generating podcast audio: {str(e)}"
        print(error_msg)
        # Return 3 seconds of silence and error message
        return (24000, np.zeros(24000 * 3)), error_msg