import os
import gradio as gr
from typing import Tuple, Dict

from voice_manager import VoiceManager

def create_voice_tab() -> Tuple[gr.Tab, VoiceManager]:
    """Create the voice management tab UI."""
    voice_manager = VoiceManager()
    
    with gr.Tab("Voice Management") as tab:
        gr.Markdown("""
        # Voice Management
        Upload audio samples to create new voices for your podcast. The audio should be:
        - Clear speech with minimal background noise
        - At least 10 seconds long
        - Single speaker
        """)
        
        with gr.Row():
            # Voice creation section
            with gr.Column():
                gr.Markdown("### Create New Voice")
                voice_name = gr.Textbox(
                    label="Voice Name",
                    placeholder="Enter a unique name for this voice"
                )
                audio_file = gr.Audio(
                    label="Voice Sample",
                    type="filepath",
                    format="wav"
                )
                with gr.Column():
                    create_btn = gr.Button("Create Voice", variant="primary", interactive=True)
                    
                    # Progress indicators
                    with gr.Row():
                        with gr.Column(scale=2):
                            create_progress = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                label="Progress",
                                interactive=False
                            )
                        with gr.Column(scale=1):
                            create_status = gr.Textbox(
                                value="Ready",
                                label="Status",
                                interactive=False
                            )
                    
                    # Detailed output
                    create_output = gr.Textbox(
                        value="",
                        label="Processing Details",
                        interactive=False,
                        lines=5
                    )
                    
                    # Audio preview after processing
                    processed_audio = gr.Audio(
                        label="Processed Audio Preview",
                        type="numpy",
                        interactive=False,
                        visible=False
                    )
            
            # Voice management section
            with gr.Column():
                gr.Markdown("### Manage Voices")
                voices_list = gr.Dropdown(
                    label="Select Voice",
                    choices=[],
                    interactive=True,
                    value=None
                )
                preview_audio = gr.Audio(
                    label="Voice Preview",
                    interactive=False
                )
                delete_btn = gr.Button("Delete Voice", variant="secondary")
                manage_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        def update_voices_list():
            """Update the list of available voices."""
            voices = voice_manager.get_available_voices()
            return {
                voices_list: gr.Dropdown(
                    choices=[{"value": v["id"], "label": f"{v['name']} ({v['type']})"} for v in voices],
                    value=None
                )
            }
        
        def create_voice(name: str, file_path: str) -> dict:
            """Handle voice creation."""
            try:
                # Input validation
                if not name or not name.strip():
                    return {
                        create_output: "Please enter a voice name",
                        create_status: "Error",
                        create_progress: 0,
                        create_btn: gr.Button(interactive=True),
                        processed_audio: gr.Audio(visible=False)
                    }
                if not file_path:
                    return {
                        create_output: "Please upload an audio file",
                        create_status: "Error",
                        create_progress: 0,
                        create_btn: gr.Button(interactive=True),
                        processed_audio: gr.Audio(visible=False)
                    }
                
                name = name.strip()
                
                # Disable button during processing
                yield {
                    create_output: "Starting voice creation process...",
                    create_status: "Processing",
                    create_progress: 0,
                    create_btn: gr.Button(interactive=False),
                    processed_audio: gr.Audio(visible=False)
                }
                
                # Process the voice
                success, message, progress_updates = voice_manager.process_audio_file(file_path, name)
                
                # Track accumulated status messages
                status_history = []
                
                # Update progress
                for progress, status in progress_updates:
                    status_history.append(status)
                    # Keep only the last 5 status messages
                    display_status = "\n".join(status_history[-5:])
                    
                    yield {
                        create_output: display_status,
                        create_status: "Processing" if progress < 100 else "Complete",
                        create_progress: progress,
                        create_btn: gr.Button(interactive=False),
                        processed_audio: gr.Audio(visible=False)
                    }
                
                if success:
                    # Update UI
                    update_voices_list()
                    
                    # Show processed audio preview
                    voice_path = voice_manager.get_voice_path(name)
                    if voice_path:
                        yield {
                            create_output: message,
                            create_status: "Complete",
                            create_progress: 100,
                            create_btn: gr.Button(interactive=True),
                            processed_audio: gr.Audio(value=voice_path, visible=True)
                        }
                    else:
                        yield {
                            create_output: message,
                            create_status: "Complete",
                            create_progress: 100,
                            create_btn: gr.Button(interactive=True),
                            processed_audio: gr.Audio(visible=False)
                        }
                else:
                    yield {
                        create_output: message,
                        create_status: "Error",
                        create_progress: 0,
                        create_btn: gr.Button(interactive=True),
                        processed_audio: gr.Audio(visible=False)
                    }
                    
            except Exception as e:
                yield {
                    create_output: f"Unexpected error: {str(e)}",
                    create_status: "Error",
                    create_progress: 0,
                    create_btn: gr.Button(interactive=True),
                    processed_audio: gr.Audio(visible=False)
                }
        
        def load_voice_preview(voice_id: str):
            """Load voice preview audio."""
            if not voice_id:
                return None
            path = voice_manager.get_voice_path(voice_id)
            return path if path else None
        
        def delete_voice(voice_id: str) -> str:
            """Handle voice deletion."""
            if not voice_id:
                return "Please select a voice to delete"
            
            success, message = voice_manager.delete_voice(voice_id)
            if success:
                # Update UI
                update_voices_list()
            return message
        
        # Wire up event handlers
        create_btn.click(
            fn=create_voice,
            inputs=[voice_name, audio_file],
            outputs=[
                create_output,
                create_status,
                create_progress,
                create_btn,
                processed_audio
            ],
            show_progress=True
        )
        
        voices_list.change(
            fn=load_voice_preview,
            inputs=[voices_list],
            outputs=[preview_audio]
        )
        
        delete_btn.click(
            fn=delete_voice,
            inputs=[voices_list],
            outputs=[manage_output]
        )
        
    return tab, voice_manager