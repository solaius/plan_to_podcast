import os
import re
import json
import torch
import datetime
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class VoiceManager:
    def __init__(self, voices_dir: str = "/workspace/voices"):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.voices_info_path = self.voices_dir / "voices_info.json"
        self.voices_info = self._load_voices_info()

    def _load_voices_info(self) -> Dict:
        """Load voice information from JSON file."""
        if self.voices_info_path.exists():
            with open(self.voices_info_path, 'r') as f:
                return json.load(f)
        return {
            "voices": {
                "default": {
                    "name": "Default Voice",
                    "description": "Default TTS voice",
                    "type": "default"
                }
            }
        }

    def _save_voices_info(self):
        """Save voice information to JSON file."""
        with open(self.voices_info_path, 'w') as f:
            json.dump(self.voices_info, f, indent=2)

    def process_audio_file(self, file_path: str, voice_name: str) -> Tuple[bool, str, List[Tuple[int, str]]]:
        """Process an audio file for voice cloning."""
        progress_updates = []
        try:
            # Input validation
            if not file_path or not os.path.exists(file_path):
                return False, "Audio file not found", []
            
            if not voice_name or not voice_name.strip():
                return False, "Voice name cannot be empty", []
            
            voice_name = voice_name.strip()
            if not re.match(r'^[a-zA-Z0-9_-]+$', voice_name):
                return False, "Voice name can only contain letters, numbers, underscores, and hyphens", []
            
            # Check if voice name already exists
            if voice_name in self.voices_info["voices"]:
                return False, "Voice name already exists", []
            
            # Load and validate audio file
            progress_updates.append((5, "Validating audio file..."))
            try:
                waveform, sample_rate = torchaudio.load(file_path)
            except Exception as e:
                return False, f"Failed to load audio file: {str(e)}", []
            
            # Basic audio validation
            if waveform.dim() == 0 or waveform.numel() == 0:
                return False, "Audio file is empty", []
            
            if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                return False, "Audio file contains invalid values", []
            
            # Validate audio length
            duration = waveform.shape[1] / sample_rate
            if duration < 5:  # At least 5 seconds
                return False, f"Audio too short ({duration:.1f}s). Please provide at least 5 seconds of audio.", []
            elif duration > 300:  # Max 5 minutes
                return False, f"Audio too long ({duration:.1f}s). Please provide audio shorter than 5 minutes.", []
            
            progress_updates.append((10, f"Loaded {duration:.1f} seconds of audio"))
            progress_updates.append((15, f"Sample rate: {sample_rate}Hz, Channels: {waveform.shape[0]}"))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                progress_updates.append((20, "Converting stereo to mono..."))
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                progress_updates.append((25, "Converted to mono successfully"))
            
            # Validate audio levels
            max_amplitude = waveform.abs().max().item()
            if max_amplitude < 0.01:
                return False, "Audio level too low. Please provide louder audio.", []
            elif max_amplitude > 1.0:
                progress_updates.append((30, "Audio levels high, will normalize..."))
            
            # Resample to 24kHz if needed
            if sample_rate != 24000:
                progress_updates.append((35, f"Resampling from {sample_rate}Hz to 24000Hz..."))
                try:
                    resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                    waveform = resampler(waveform)
                    progress_updates.append((40, "Resampling completed successfully"))
                except Exception as e:
                    return False, f"Failed to resample audio: {str(e)}", []
            
            # Normalize audio
            progress_updates.append((50, "Normalizing audio..."))
            waveform = waveform / waveform.abs().max()
            
            # Trim silence using a more robust method
            progress_updates.append((60, "Trimming silence..."))
            try:
                # Convert to numpy for more reliable silence detection
                audio_np = waveform.numpy()
                
                # Calculate RMS energy
                frame_length = 1024
                hop_length = 512
                rms = np.sqrt(np.mean(np.square(
                    audio_np.reshape(-1, frame_length)), 
                    axis=1
                ))
                
                # Find non-silent parts
                threshold = 0.01 * np.max(rms)
                non_silent_frames = rms > threshold
                
                if np.any(non_silent_frames):
                    # Find first and last non-silent frame
                    first_frame = np.argmax(non_silent_frames)
                    last_frame = len(non_silent_frames) - np.argmax(non_silent_frames[::-1]) - 1
                    
                    # Convert frame indices to sample indices
                    start_sample = first_frame * hop_length
                    end_sample = min((last_frame + 1) * hop_length, audio_np.shape[1])
                    
                    # Trim the waveform
                    waveform = torch.from_numpy(
                        audio_np[:, start_sample:end_sample]
                    ).to(waveform.device)
                    
                    progress_updates.append((65, f"Trimmed {(start_sample/audio_np.shape[1]*100):.1f}% from start, {((audio_np.shape[1]-end_sample)/audio_np.shape[1]*100):.1f}% from end"))
                else:
                    progress_updates.append((65, "No non-silent parts found, using full audio"))
            except Exception as e:
                progress_updates.append((60, f"Warning: Silence trimming failed: {str(e)}. Using full audio."))
            
            # Save processed audio
            progress_updates.append((80, "Saving processed audio..."))
            voice_path = self.voices_dir / f"{voice_name}.wav"
            
            # Ensure voices directory exists
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the audio file
            try:
                torchaudio.save(voice_path, waveform, 24000)
                progress_updates.append((85, f"Audio saved to {voice_path.name}"))
            except Exception as e:
                return False, f"Failed to save audio file: {str(e)}", progress_updates
            
            # Verify the saved file
            try:
                test_load, test_sr = torchaudio.load(voice_path)
                if test_sr != 24000 or test_load.shape != waveform.shape:
                    raise ValueError("Saved audio file verification failed")
                progress_updates.append((87, "Verified saved audio file"))
            except Exception as e:
                # Clean up failed file
                if voice_path.exists():
                    voice_path.unlink()
                return False, f"Audio file verification failed: {str(e)}", progress_updates
            
            # Calculate audio statistics
            stats = {
                "duration": duration,
                "sample_rate": 24000,
                "channels": waveform.shape[0],
                "peak_amplitude": float(waveform.abs().max()),
                "rms_level": float(torch.sqrt(torch.mean(waveform ** 2))),
            }
            
            # Update voice info
            progress_updates.append((90, "Updating voice database..."))
            voice_info = {
                "name": voice_name,
                "description": f"Cloned voice from {os.path.basename(file_path)}",
                "type": "cloned",
                "path": str(voice_path),
                "stats": stats,
                "created": str(datetime.datetime.now()),
                "source_file": os.path.basename(file_path)
            }
            
            self.voices_info["voices"][voice_name] = voice_info
            
            try:
                self._save_voices_info()
                progress_updates.append((95, "Voice database updated"))
            except Exception as e:
                # Clean up on database save failure
                if voice_path.exists():
                    voice_path.unlink()
                return False, f"Failed to update voice database: {str(e)}", progress_updates
            
            # Success message with stats
            success_msg = (
                f"Voice '{voice_name}' created successfully!\n"
                f"Duration: {duration:.1f}s\n"
                f"Sample Rate: 24kHz\n"
                f"Channels: {waveform.shape[0]}\n"
                f"Peak Level: {20 * np.log10(stats['peak_amplitude']):.1f}dB\n"
                f"RMS Level: {20 * np.log10(stats['rms_level']):.1f}dB"
            )
            
            progress_updates.append((100, "Voice processing complete!"))
            return True, success_msg, progress_updates
            
        except Exception as e:
            # Clean up any partial files
            if 'voice_path' in locals() and voice_path.exists():
                voice_path.unlink()
            
            error_msg = f"Unexpected error: {str(e)}"
            progress_updates.append((0, error_msg))
            return False, error_msg, progress_updates

    def get_available_voices(self) -> List[Dict]:
        """Get list of available voices."""
        voices = []
        
        # Always include default voice first
        if "default" in self.voices_info["voices"]:
            voices.append({
                "id": "default",
                "name": self.voices_info["voices"]["default"]["name"],
                "type": "default"
            })
        
        # Add other voices
        for vid, info in self.voices_info["voices"].items():
            if vid != "default":
                voices.append({
                    "id": vid,
                    "name": info["name"],
                    "type": info["type"]
                })
        
        return voices

    def get_voice_path(self, voice_id: str) -> Optional[str]:
        """Get the path to a voice's audio file."""
        if voice_id in self.voices_info["voices"]:
            return self.voices_info["voices"][voice_id].get("path")
        return None

    def delete_voice(self, voice_id: str) -> Tuple[bool, str]:
        """Delete a cloned voice."""
        try:
            if voice_id not in self.voices_info["voices"]:
                return False, "Voice not found"
            
            if self.voices_info["voices"][voice_id]["type"] == "default":
                return False, "Cannot delete default voice"
            
            # Delete audio file if it exists
            voice_path = self.voices_info["voices"][voice_id].get("path")
            if voice_path:
                try:
                    os.remove(voice_path)
                except:
                    pass
            
            # Remove from info
            del self.voices_info["voices"][voice_id]
            self._save_voices_info()
            
            return True, "Voice deleted successfully"
            
        except Exception as e:
            return False, f"Error deleting voice: {str(e)}"