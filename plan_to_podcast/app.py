import gradio as gr

from plan_to_podcast.constants import BANNER_TEXT, EXAMPLES
from plan_to_podcast.generate_podcast import generate_podcast_script
from plan_to_podcast.step_tts import podcast_tts
from plan_to_podcast.utils import get_models
from plan_to_podcast.voice_ui import create_voice_tab
from plan_to_podcast.voice_manager import VoiceManager

MODELS = get_models()
voice_manager = VoiceManager()

def get_voice_choices():
    """Get list of available voices for dropdowns."""
    voices = voice_manager.get_available_voices()
    return [{"value": v["id"], "label": f"{v['name']} ({v['type']})"} for v in voices]

# Main application
with gr.Blocks() as demo:
    with gr.Tabs():
        # Podcast Generation Tab
        with gr.Tab("Generate Podcast"):
            with gr.Row():
                gr.Markdown(BANNER_TEXT)
            with gr.Row(variant="panel"):
                with gr.Accordion("0. (Optional) Select a preloaded script", open=False):
                    example = gr.Dropdown(label="Example", choices=list(EXAMPLES.keys()), interactive=True)
                    load_example = gr.Button(value="Load Example", variant="secondary")

            with gr.Row():
                # Script Generation Column
                with gr.Column(variant="panel"):
                    gr.Markdown("## 1. Generate a Podcast Script")
                    with gr.Row():
                        host_a = gr.Textbox(label="Host 1 Name", value="Lily")
                        host_b = gr.Textbox(label="Host 2 Name", value="Marshall")
                    with gr.Row():
                        # Get voice choices
                        voice_choices = get_voice_choices()
                        
                        # Create voice dropdowns with allow_custom_value=True to prevent warnings
                        voice_a = gr.Dropdown(
                            choices=voice_choices,
                            value="default",
                            label="Host 1 Voice",
                            interactive=True,
                            allow_custom_value=True
                        )
                        voice_b = gr.Dropdown(
                            choices=voice_choices,
                            value="default",
                            label="Host 2 Voice",
                            interactive=True,
                            allow_custom_value=True
                        )
                        host_voices = {host_a.value: voice_a.value, host_b.value: voice_b.value}
                        host_voices = gr.JSON(value=host_voices, visible=False)
                    with gr.Row():
                        topic = gr.Textbox(
                            label="Topic",
                            info="Topic for your podcast. You can also specify key points for the hosts to talk about.",
                            max_lines=5
                        )
                    with gr.Row():
                        default_model = "qwen2.5:32b" if "qwen2.5:32b" in MODELS else MODELS[0]
                        model = gr.Dropdown(
                            MODELS,
                            value=default_model,
                            label="Generation Model",
                            info="LLM to use for generating script"
                        )
                    with gr.Row():
                        generate_btn = gr.Button("Generate", variant="primary")
                
                # TTS column
                with gr.Column(variant="panel"):
                    gr.Markdown("## 2. Review Generated Script")
                    script = gr.Textbox(
                        show_label=False,
                        max_lines=5,
                        info="Podcast script. Each conversation turn must in the format '<|speaker|>: content', separated by a blank line.",
                    )
                    gr.Markdown("## 3. Convert Podcast Script to Audio")

                    with gr.Row():
                        tts_btn = gr.Button("Generate", variant="primary")
                    with gr.Row():
                        out_audio = gr.Audio(label="Output Audio", interactive=False, streaming=False, autoplay=True)
                    with gr.Accordion("Output Tokens", open=False):
                        out_ps = gr.Textbox(interactive=False, show_label=False, info="Tokens used to generate the audio.", lines=15)

            # Event handlers for podcast generation
            def update_host_voices():
                return gr.JSON(value={host_a.value: voice_a.value, host_b.value: voice_b.value})

            load_example.click(
                lambda x: (EXAMPLES[x]["topic"], EXAMPLES[x]["script"].strip()),
                inputs=[example],
                outputs=[topic, script]
            )
            
            generate_btn.click(
                fn=generate_podcast_script,
                inputs=[topic, model, host_a, host_b],
                outputs=[script]
            )
            
            # Update host_voices when any related field changes
            host_a.change(fn=update_host_voices, inputs=[], outputs=[host_voices])
            host_b.change(fn=update_host_voices, inputs=[], outputs=[host_voices])
            voice_a.change(fn=update_host_voices, inputs=[], outputs=[host_voices])
            voice_b.change(fn=update_host_voices, inputs=[], outputs=[host_voices])
            
            tts_btn.click(
                fn=podcast_tts,
                inputs=[script, host_voices],
                outputs=[out_audio, out_ps]
            )
        
        # Voice Management Tab
        voice_tab, _ = create_voice_tab()

demo.launch(server_name="0.0.0.0", server_port=52881)
