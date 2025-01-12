import streamlit as st
from utils import get_translation, get_image_prompts, segments_to_chunks, generate_images, generate_video
import constants  
from groq import Groq

client = Groq()

# Initialize state variables if not already set
if 'transcript_visible' not in st.session_state:
    st.session_state.transcript_visible = False
if 'translation_visible' not in st.session_state:
    st.session_state.translation_visible = False
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'audio' not in st.session_state:
    st.session_state.audio = None
if 'was_converted' not in st.session_state:
    st.session_state.was_converted = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'translation' not in st.session_state:
    st.session_state.translation = None
if 'generated_video' not in st.session_state:
    st.session_state.generated_video = None
if 'image_prompts' not in st.session_state:
    st.session_state.image_prompts = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = None


# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>AI Video Generator</h1>",
    unsafe_allow_html=True
)
st.info("Video Generation Feature - Functional But Can be Buggy")

# Upload audio file
audio_file = st.file_uploader("🔼 Upload your audio file:", type=constants.SUPPORTED_FORMATS)

print(audio_file,'is the upload')

# if audio_file is not None:
#     # Check the duration of the uploaded audio file
#     duration = get_audio_duration(audio_file)

#     # Allow only files up to 5 minutes (300 seconds)
#     if duration > 300:
#         st.error("The uploaded audio file exceeds the 5-minute limit. Please upload a shorter file.")
#     else:
#         st.success(f"Audio file uploaded successfully! Duration: {duration/60:.2f} minutes")

if audio_file:
    # Reset states only when a new file is uploaded
    if st.session_state.uploaded_file_name != audio_file.name:
        st.session_state.uploaded_file_name = audio_file.name
        st.session_state.audio = audio_file
        st.session_state.transcript = None
        st.session_state.translation = None
        st.session_state.image_prompts = None
        st.session_state.generated_images = None  # Reset image generation state

    st.info(f"Uploaded file: **{audio_file.name}**")

    # Read the uploaded file's bytes and send to Groq API for transcription
    file_bytes = audio_file.read()

    # Create a transcription of the audio file using Groq API
    result = client.audio.transcriptions.create(
        file=(audio_file.name, file_bytes),  # Send the audio file content directly to the API
        model="whisper-large-v3-turbo",  # Model to use for transcription
        prompt="Specify context or spelling",  # Optional context for better transcription accuracy
        response_format="verbose_json",  # Return detailed JSON response
        temperature=0.0,  # Control randomness in the transcription output
    )
    st.session_state.transcript = result.text
    st.session_state.segments = result.segments

    # Translation logic
    if st.session_state.transcript and st.session_state.translation is None:
        with st.spinner("Generating translation... Please wait."):
            st.session_state.translation = get_translation(st.session_state.transcript)

    st.audio(st.session_state.audio, format=f"audio/{audio_file.type}")

    # Toggle transcript visibility
    toggle_transcript = st.checkbox("Show Transcript", value=st.session_state.transcript_visible, key="toggle_transcript")
    st.session_state.transcript_visible = toggle_transcript

    if st.session_state.transcript_visible:
        st.write("### Transcription:")
        st.write(st.session_state.transcript)

    # Toggle translation visibility
    toggle_translation = st.checkbox("Show Translation", value=st.session_state.translation_visible, key="toggle_translation")
    st.session_state.translation_visible = toggle_translation

    if st.session_state.translation_visible:
        st.write("### Translation:")
        st.write(st.session_state.translation)

    # Image generation logic
    if st.session_state.translation and st.session_state.image_prompts is None:
        with st.spinner("Generating image prompts... Please wait."):
            if 'Already in English' in st.session_state.translation:
                st.info("Audio is Already in English. Using Transcription to generate Image Prompts")
                st.session_state.image_prompts = get_image_prompts(segments_to_chunks(st.session_state.segments))['image_prompts']
            else:
                st.session_state.image_prompts = get_image_prompts(segments_to_chunks(st.session_state.segments))['image_prompts']

    print(st.session_state.image_prompts)
    # Ensure that generated_images is always a list
    if 'generated_images' not in st.session_state or st.session_state.generated_images is None:
        st.session_state.generated_images = []

    # Generate images only if they have not been generated already
    if st.session_state.image_prompts and not st.session_state.generated_images:
        with st.spinner("Generating images... Please wait."):
            for prompt, image_path in generate_images(st.session_state.image_prompts):
                # # Display each image as soon as it's generated
                # st.image(image_path, caption=f"{prompt}", use_container_width=True)
                # Append the generated image to the session state
                st.session_state.generated_images.append((prompt, image_path))

    # # Display all previously generated images (including newly generated ones)
    # else:
    #     for prompt, image_path in st.session_state.generated_images:
    #         st.image(image_path, caption=f"{prompt}", use_container_width=True)

    # Generate video when all images are generated
    if st.session_state.generated_images and st.session_state.audio:
        if st.button("Generate Video"):
            with st.spinner("Generating video... Please wait."):
                # Map images to segments
                image_paths = [img[1] for img in st.session_state.generated_images]
                generated_video_path = generate_video(
                    audio_file=st.session_state.audio, 
                    images=image_paths, 
                    segments=st.session_state.segments
                )
                st.session_state.generated_video = generated_video_path
                st.success("Video generated successfully!")

    # Display the generated video
    if st.session_state.generated_video:
        st.video(st.session_state.generated_video)

else:
    st.warning("Please upload an audio file to proceed.")


