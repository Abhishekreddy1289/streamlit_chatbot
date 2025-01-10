import time
import streamlit as st

def cache_answer(answer):
    answer_list = answer.split()

    # Function to stream the answer as text
    def stream_answer():
        for i in answer_list:
            yield i + " "
            time.sleep(0.02)

    # Display the answer progressively
    st.write_stream(stream_answer)

    # Function to play bot audio
    def play_bot_audio(answer):
        st.write(f"Playing audio for: {answer}")  # Display audio play debug message
        # Add the actual logic to play the audio here.

    # Create a button to trigger the audio playback
    if st.button('Play Audio'):
        st.write("*" * 100)  # Show a message when the button is pressed
        play_bot_audio(answer)  # This should call the function when the button is pressed

# Sample usage of the cache_answer function
answer = "Hello, how can I assist you today?"
cache_answer(answer)
