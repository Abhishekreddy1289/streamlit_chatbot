import os
import time
import asyncio
from io import BytesIO
import streamlit as st
from rephrase import Rephrase
from rag_pipeline import RAG, TextProcessor
from audio_tts import TextToSpeech  # Importing the TextToSpeech class
import time
import sys
from io import StringIO

# Simulate the user entering "y"
sys.stdin = StringIO('y\n')

os.environ['COQUI_LICENSE_ACCEPTED'] = 'y'

tts = TextToSpeech() 
#Initialize TextProcessor class
textprocessor=TextProcessor(chunk_size=1000, chunk_overlap=20)

# # Function to convert bot answer to speech and stream it
# def play_bot_audio(answer):
#     audio_file = "response.wav"
#     tts.text_to_speech(text=answer, file_path=audio_file)  # Generate audio from text
#     st.audio(audio_file) 


# Function to convert bot answer to speech and stream it
def play_bot_audio(answer):
    # Define the max length of each chunk (250 characters)
    max_chunk_length = 200
    
    # Split the answer into chunks without breaking words
    chunks = []
    while len(answer) > max_chunk_length:
        # Find the last space within the 250-character limit
        split_point = answer.rfind(' ', 0, max_chunk_length)
        if split_point == -1:  # If no space is found (edge case)
            split_point = max_chunk_length  # Force break at the max length
        chunks.append(answer[:split_point])  # Append the chunk
        answer = answer[split_point:].lstrip()  # Remove the chunk from the original answer

    # Append the remaining part of the answer (less than 250 characters)
    if answer:
        chunks.append(answer)
    print("chunks size:-",len(chunks))
    # # Play each chunk sequentially
    # for chunk in chunks:
    #     audio_file = "response.wav"
    #     tts.text_to_speech(text=chunk, file_path=audio_file)  # Generate audio from text for the chunk
    #     st.audio(audio_file)  # Play the audio
    #     time.sleep(1)  # Small delay between chunks to ensure smooth transition
     # Generate and save audio for each chunk
    audio_files=[]
    for i, chunk in enumerate(chunks):
        audio_file = f"response_{i}.wav"
        tts.text_to_speech(text=chunk, file_path=audio_file)  # Generate audio from text for the chunk
        audio_files.append(audio_file)  # Add the file to the list

    # Combine all audio chunks into one
    combined_audio = AudioSegment.empty()  # Start with an empty audio segment
    for audio_file in audio_files:
        audio_segment = AudioSegment.from_wav(audio_file)  # Load each audio file
        combined_audio += audio_segment  # Append to the combined audio

    # Export the combined audio to a single file
    combined_audio_file = "combined_response.wav"
    combined_audio.export(combined_audio_file, format="wav")

    # Play the combined audio file
    st.audio(combined_audio_file)

    # Clean up the individual chunk audio files (optional)
    for audio_file in audio_files:
        os.remove(audio_file)


os.environ['PINECONE_API_KEY'] = 'pcsk_4hp8P_FGu8ZMCRX7gEjF1AH7ZTF3fv6V4uFyrTpNHkCidkvCofdF6LjR4QLwSZCpqSTGC'
PINECONE_API_KEY="pcsk_4hp8P_FGu8ZMCRX7gEjF1AH7ZTF3fv6V4uFyrTpNHkCidkvCofdF6LjR4QLwSZCpqSTGC"


# List of recommended questions
recommended_questions = [
    "What is the refund process for failed reservations, and which charges may be forfeited?",
    "How are cancellation charges calculated for confirmed tickets based on time and class?",
    "How is a refund processed for a party e-ticket with mixed confirmed and RAC/waitlisted passengers?"
]

answers=['''For failed reservations, if the amount is debited but the ticket is not issued, IRCTC will refund the full fare and convenience fee electronically to the payment account. However, bank/card transaction charges may be forfeited. Refunds are processed electronically but may be delayed due to multiple organizations involved in transaction processing.''',
         '''Cancellation charges for confirmed tickets are calculated based on the time of cancellation and the class of the ticket. The charges are as follows:

1. **More than 48 hours before departure:**  
   - AC First/Executive Class: Rs. 240 + GST  
   - First Class/AC 2 Tier: Rs. 200 + GST  
   - AC Chair Car/AC 3 Tier/AC 3 Economy: Rs. 180 + GST  
   - Sleeper Class: Rs. 120  
   - Second Class: Rs. 60  

2. **Between 48 hours and 12 hours before departure:**  
   - 25% of the fare, subject to a minimum of the cancellation charges mentioned above, plus GST.

3. **Within 12 hours to 4 hours before departure:**  
   - 50% of the fare, subject to a minimum of the cancellation charges mentioned above, plus GST.

4. **No refund is given if canceled less than 4 hours before departure.**''',
'''For a party e-ticket with mixed confirmed and RAC/waitlisted passengers:

- If the entire ticket is canceled before 30 minutes of the scheduled departure, a full refund (minus clerkage) is granted for the confirmed passengers.
- If some passengers on the ticket are RAC or waitlisted and do not travel, a certificate from the ticket checking staff is required. The refund will be processed online via TDR, and the TDR must be filed within 72 hours of the train's actual arrival at the destination. The original certificate must be sent by post to IRCTC for processing.''']

# Set the page title and layout
st.set_page_config(page_title="Chatbot Interface", layout="wide")

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'endpoint' not in st.session_state:
    st.session_state.endpoint = ''
if 'version' not in st.session_state:
    st.session_state.version = ''
if 'model_name' not in st.session_state:
    st.session_state.model_name = ''
if 'embedding_model_name' not in st.session_state:
    st.session_state.embedding_model_name = ''
if 'openai_type' not in st.session_state:
    st.session_state.openai_type = ''
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Function to display the modal
def display_modal():
    with st.expander("Please enter your Model Details:", expanded=False):
        # User selection for OpenAI or Azure OpenAI
        service = st.selectbox("Select Service", ["OpenAI", "Azure OpenAI"])
        with st.form(key='model_details_form'):
            if service == "OpenAI":
                st.session_state.openai_type = "openai"
                st.session_state.api_key = st.text_input("API Key", type="password")
                st.session_state.model_name = st.text_input("Model Name")
                st.session_state.embedding_model_name = st.text_input("Embedding Model Name")
                st.warning("Use text-embedding-002 model as VectorDB is optimized for it.")
            elif service == "Azure OpenAI":
                st.session_state.openai_type = "azure_openai"
                st.session_state.api_key = st.text_input("API Key", type="password")
                st.session_state.endpoint = st.text_input("Endpoint")
                st.session_state.version = st.text_input("Version")
                st.session_state.model_name = st.text_input("Model Name")
                st.session_state.embedding_model_name = st.text_input("Embedding Model Name")
                st.warning("Use text-embedding-002 model as VectorDB is optimized for it.")

            if st.form_submit_button("Submit"):
                # Collect and display the input values
                st.success("Details submitted successfully!")

# Display the modal when the app starts
display_modal()

# Function to show notification
def show_notification(message, message_type="success"):
    # Define CSS styles for the notification
    css = f"""
    <style>
    .notification {{
        position: fixed;
        top: 0;
        right: 0;
        margin: 20px;
        padding: 10px 20px;
        color: white; /* Text color */
        border-radius: 5px;
        z-index: 1000; /* Ensure it's on top */
        background-color: {"#4CAF50" if message_type == "success" else "#f44336"}; /* Green for success, red for error */
    }}
    </style>
    """
    notification_placeholder = st.empty()
    notification_placeholder.markdown(f'{css}<div class="notification">{message}</div>', unsafe_allow_html=True)
    time.sleep(5)
    notification_placeholder.empty()


if st.session_state.api_key:
    openai_type=st.session_state.openai_type
    gpt_engine_name=st.session_state.model_name
    embedding_model_name=st.session_state.embedding_model_name
    azure_endpoint = st.session_state.endpoint
    api_key=st.session_state.api_key
    api_version= st.session_state.version
    rephrase_obj=Rephrase(gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type)
    rag_obj=RAG("indexdb1", textprocessor, PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type)


# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

uploaded_files = None
selected_file = None


# Handle file uploads only when user interacts with the sidebar
with st.sidebar:
    uploaded_files = st.file_uploader('Upload files', type=['pdf'], accept_multiple_files=True, label_visibility="hidden")

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            if uploaded_file not in st.session_state.uploaded_files:
                bytes_data = uploaded_file.read()
                file_like_object = BytesIO(bytes_data)
                file_name = uploaded_file.name
                st.session_state.uploaded_files.append(uploaded_file)
                
                try:
                    if st.session_state.api_key:
                        show_notification(f"{uploaded_file.name} preprocessing started, Please wait for a while!")
                        if file_name.endswith(('.pdf')):
                            rag_obj.insert_doc(file_like_object, file_name)
                        else:
                            graph_obj.insert_docs(file_like_object, file_name)
                        show_notification(f"{uploaded_file.name} inserted successfully!")
                    else:
                        show_notification("Please enter your OpenAI credentials before uploading documents!", message_type='error')
                except Exception as e:
                    show_notification("Something went wrong while preprocessing, please try again!", message_type='error')
    
    st.write("Select files:")
    files_to_keep = []
    for uploaded_file in st.session_state.uploaded_files:
        is_checked = st.checkbox(uploaded_file.name, value=uploaded_file in st.session_state.selected_files)
        if is_checked:
            files_to_keep.append(uploaded_file)
            if uploaded_file not in st.session_state.selected_files:
                st.session_state.selected_files.append(uploaded_file)
        else:
            if uploaded_file in st.session_state.selected_files:
                st.session_state.selected_files.remove(uploaded_file)

    # Update the session state with the currently selected files
    st.session_state.selected_files = files_to_keep


# Main content area
st.markdown("### What can I help you with today?")

# Display the message
st.write("You can upload documents in the sidebar")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state for the selected question
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = "What is up?"

def handle_button_click(question):
    st.session_state.selected_question = question

# Function to handle button click
def handle_button_click(question):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.selected_question = question
    st.session_state.messages.append({"role": "user", "content": question})
    # Trigger bot query
    for i, j in enumerate(recommended_questions):
            if j==question:
                answer=answers[i]
    cache_answer(answer)
    st.warning("The answer is from the cache!")

def cache_answer(answer):
    answer_list=answer.split()
    def stream_answer():
        for i in answer_list:
            yield i + " "
            time.sleep(0.02)
    st.write_stream(stream_answer)
    st.session_state.messages.append({"role": "Bot", "content": answer})
    play_bot_audio(answer)


def process_input(prompt, rephrase_prompt):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    selected_files=st.session_state.selected_files
    selected_names=[]
    for file in selected_files:
        selected_names.append(file.name)
    print("__________________________________________________________ selected files", selected_names)
    filters={
  "filename": { "$in": selected_names }
}
    print("__________________________________________________________ filtered files", filters)
    try:
        if st.session_state.api_key:
            answer=rag_obj.qna(rephrase_prompt, filters)
            with st.chat_message("BOT"):
                st.write_stream(answer)
            # Add user message to chat history
            final_answer=rag_obj.answer
            st.session_state.messages.append({"role": "Bot", "content": final_answer})
            rag_obj.answer=''
            play_bot_audio(final_answer)
        else:
            st.error("Please enter your API Key and other details at the top.")
    except Exception as e:
        print("error", e)
        answer="Something went wrong, please try again(also check your openai credentials)."
        st.error(answer)


# Display recommended questions as buttons
if not st.session_state.messages:
    st.write("**Recommended Questions:**")
    cols = st.columns(len(recommended_questions))
    for i, question in enumerate(recommended_questions):
        if cols[i].button(question):
            handle_button_click(question)

# Prefill input bar with the selected question if available
if query := st.chat_input(st.session_state.selected_question):
        print("User Query:",query)
        history=""
        for i in st.session_state.messages[-2:]:
            if i['role']=='user':
                history+=f"User: {i['content']}\n"
            if i['role']=='Bot':
                history+=f"Bot: {i['content']}\n"
        rephrase_prompt=rephrase_obj.followup_query(query, history)
        print("Rephrase Query:",rephrase_prompt)
        process_input(query, rephrase_prompt)


# Dummy element to trigger auto-scrolling
auto_scroll = st.empty()

# Trigger auto-scroll by adding an empty message after all other messages
with auto_scroll:
    st.write("")  # This empty write forces the page to render and auto-scroll