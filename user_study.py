import os
import io
import streamlit as st
import streamlit.components.v1 as components
import random
import csv
import base64
import uuid
import time
import datetime
import json
import pandas as pd

from st_files_connection import FilesConnection

# --- Configuration ---
VIDEO_BASE_PATH = "videos/test"
INPUT_VIDEO_PATH = "davis_eval_proc"
OUR_METHOD_NAME = "prune0.149" 
GCS_BUCKET_NAME = "streamlit-human-eval-bucket"
conn = st.connection('gcs', type=FilesConnection)

BASELINE_FOLDERS = {
    "no_prune": "no_prune",
    # "streamv2v": "StreamV2V_Processed_Outputs",
    # "streamdiffusion": "StreamDiffusion_Processed_Outputs",
    # "controlvideo": "ControlVideo_Outputs"
}

PROMPT_PATH = "eval_prune.json"
with open(PROMPT_PATH, 'r') as f:
        prompts_json = json.load(f) 

# --- Helper Functions ---
def find_matching_prompt(file_name, davis_prompts):
    """
    Searches through davis_data to find if parsed_key is a substring 
    of any 'edit_vid' prompt. Returns the full prompt if found.
    """
    for entry in davis_prompts:
        full_prompt = entry.get('edit_vid', '')
        if file_name in full_prompt.replace('-', ' '):
            return full_prompt
            
    return None

def render_video_iframe(filepath, placeholder_obj):
    """
    Renders a video inside a sandboxed Iframe using Base64.
    Args:
        filepath: Path to the video file.
        placeholder_obj: A st.empty() container to render into.
    """
    # 1. Debugging print
    print(f"DEBUG: Loading video from {filepath}")
    
    if not os.path.exists(filepath):
        placeholder_obj.error(f"Video not found: {filepath}")
        return

    try:
        # 2. Read file
        with open(filepath, "rb") as f:
            video_bytes = f.read()
            print(f"DEBUG: Read {len(video_bytes)} bytes.")
    except Exception as e:
        placeholder_obj.error(f"Error reading video: {e}")
        return

    # 3. Base64 Encode
    b64 = base64.b64encode(video_bytes).decode()
    
    # 4. Generate Unique ID to force browser refresh
    unique_id = f"vid_{uuid.uuid4()}"
    
    # 5. Build HTML (Iframe content)
    # We set height/width on the video tag to fit the iframe
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0; padding:0; background-color: black;">
        <video id="{unique_id}" width="100%" height="100%" autoplay loop muted controls playsinline>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </body>
    </html>
    """
    
    # 6. Render the Iframe
    with placeholder_obj:
        components.html(html_content, height=320, width=555, scrolling=False)

def get_comparison_pairs(group_id, seed=0):
    pairs = []
    ours_path_root = os.path.join(VIDEO_BASE_PATH, OUR_METHOD_NAME)
    
    if not os.path.exists(ours_path_root):
        st.error(f"⚠️ Could not find folder: {ours_path_root}")
        return []

    all_videos = [f for f in sorted(os.listdir(ours_path_root)) if f.endswith(".mp4")]

    if len(all_videos) == 0:
        st.error(f"⚠️ No .mp4 files found in folder: {ours_path_root}")
        return []

    rng = random.Random(seed)
    rng.shuffle(all_videos)
    
    mid_point = len(all_videos) // 2
    
    selected_videos = []
    if group_id == 1:
        selected_videos = all_videos[:mid_point]
    elif group_id == 2:
        selected_videos = all_videos[mid_point:]
    elif group_id == 3:
        selected_videos = all_videos
    else:
        st.error("Invalid Test ID provided.")
        return []

    for video_file in selected_videos:
        caption = video_file.replace(".mp4", "")
        path_to_ours = os.path.abspath(os.path.join(ours_path_root, video_file))
        path_to_source = os.path.abspath(os.path.join(VIDEO_BASE_PATH, INPUT_VIDEO_PATH, video_file))
        for baseline_key, baseline_folder in BASELINE_FOLDERS.items():
            baseline_path_root = os.path.join(VIDEO_BASE_PATH, baseline_folder)
            path_to_baseline = os.path.join(baseline_path_root, video_file)

            if os.path.exists(path_to_baseline):
                pairs.append({
                    "caption": caption,
                    "filename": video_file,
                    "source_path": path_to_source,
                    "ours_path": path_to_ours,
                    "baseline_name": baseline_key,
                    "baseline_path": os.path.abspath(path_to_baseline)
                })
    
    pairs.sort(key=lambda x: x['filename'])
    return pairs

# def save_result(caption, baseline_method, chosen_method, file_name):
    parent_folder = os.path.dirname(file_name)

    if parent_folder:  
        os.makedirs(parent_folder, exist_ok=True)

    with open(file_name, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["Caption", "Baseline_Method", "Chosen_Method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        file_exists = os.path.exists(file_name)
        if not file_exists: writer.writeheader()
        writer.writerow({
            "Caption": caption,
            "Baseline_Method": baseline_method,
            "Chosen_Method": chosen_method
        })

def save_result(caption, baseline_method, chosen_method, file_name):
    full_gcs_path = f"{GCS_BUCKET_NAME}/{file_name}"
    
    # 1. Create the new row as a DataFrame
    new_data = pd.DataFrame([{
        "Caption": caption,
        "Baseline_Method": baseline_method,
        "Chosen_Method": chosen_method
    }])

    # 2. Try to read the existing CSV
    try:
        # CRITICAL: ttl=0 ensures we fetch the latest file, not a cached version
        existing_df = conn.read(full_gcs_path, input_format="csv", ttl=0)
        
        # Concatenate old data with new data
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        
    except Exception:
        # If file not found (or empty), start fresh with just the new data
        updated_df = new_data

    # 3. Write the combined DataFrame back to GCS
    try:
        # We still use conn.open with 'wt' (write text) to save the file
        with conn.open(full_gcs_path, mode='wt') as f:
            updated_df.to_csv(f, index=False)
    except Exception as e:
        st.error(f"Failed to save to GCS: {e}")

def show_landing_page():
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

    # 1. Define constants at the top for cleaner logic
    # Mapping user input strings to internal IDs
    ID_MAPPING = {
        "1111": 1,
        "2222": 2,
        "all": 3
    }
    
    INSTRUCTIONS = """
    ### Instructions
    :red[Please read the instructions carefully before starting.]

    We use AI models to edit source videos based on provided prompts. Given two edited videos, please select the one you prefer.

    Your judgment should base on the following criteria:

    * **Edited Video Quality:** e.g. Smoothness of frames, consistency of content, and per frame (image) quality.
    * **Prompt Alignment:** How well the edited video matches the given prompt.
    * **Motion Fidelity:** How well the motion matches the source video.

    **Note:** If these factors contradict each other, please **prioritize video quality**, as the result must be a functional video first.

    After reviewing both videos, select your preferred option or choose **"Draw"** if you cannot decide which one is better.

    Please use a desktop or laptop to complete this evaluation.

    Click **"Start Evaluation"** to begin.
    """

    # 2. Render UI Elements
    st.markdown(INSTRUCTIONS)
    
    # Using a form creates a better UI feel (users can hit Enter or click the button)
    with st.form("login_form"):
        user_input = st.text_input("Enter your test ID:")
        user_name = st.text_input("Enter your name")
        submitted = st.form_submit_button("Start Evaluation")

        if submitted:
            # 3. specific logic using Dictionary Lookup
            # .strip() removes accidental spaces
            clean_input = user_input.strip()

            if clean_input in ID_MAPPING and user_name.strip() != "":
                st.session_state.test_id = ID_MAPPING[clean_input]
                st.session_state.user_name = user_name.strip()
                st.session_state.page_index = 1
                st.rerun()
            else:
                st.error("Invalid Test ID or Name. Please try again.")

# --- Main App ---

def main():
    st.set_page_config(layout="wide", page_title="Video Eval")
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

    if st.session_state.get('page_index', 0) == 0:
        show_landing_page()
        return
    
    st.title("Video Comparison Study")   

    if "user_session_id" not in st.session_state:
        timestamp = time.time()
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        readable_date = dt_object.strftime("%Y-%m-%d_%H-%M-%S")
        user_name = st.session_state.get('user_name', '').replace(" ", "_")
        st.session_state.user_session_id = f"{readable_date}_{user_name}"

    if "task_index" not in st.session_state:
        st.session_state.task_index = 0

    all_pairs = get_comparison_pairs(st.session_state.test_id)
    if not all_pairs:
        st.stop()
    
    if st.session_state.task_index >= len(all_pairs):
        st.success(f"🎉 Session complete! You have finished {len(all_pairs)} evaluations.")
        if st.button("Start Over"):
            st.session_state.task_index = 0
            if "user_session_id" in st.session_state:
                del st.session_state["user_session_id"]
            if "page_index" in st.session_state:
                del st.session_state["page_index"]
            if "test_id" in st.session_state:
                del st.session_state["test_id"]
            if "user_name" in st.session_state:
                del st.session_state["user_name"]
            st.rerun()
        return

    current_pair = all_pairs[st.session_state.task_index]
    current_session_id = f"{current_pair['caption']}_{current_pair['baseline_name']}"

    if "last_session_id" not in st.session_state or st.session_state.last_session_id != current_session_id:
        st.session_state.last_session_id = current_session_id
        st.session_state.ours_is_A = random.choice([True, False]) 

    ours_is_A = st.session_state.ours_is_A

    if ours_is_A:
        path_a, name_a = current_pair['ours_path'], OUR_METHOD_NAME
        path_b, name_b = current_pair['baseline_path'], current_pair['baseline_name']
    else:
        path_a, name_a = current_pair['baseline_path'], current_pair['baseline_name']
        path_b, name_b = current_pair['ours_path'], OUR_METHOD_NAME

    # --- Comparison Videos ---
    prompt = find_matching_prompt(current_pair['filename'].removesuffix(".mp4"), prompts_json)
    st.markdown(f"#### Prompt: {prompt}")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Source Video")
        ph_source = st.empty()

    with col2:
        st.subheader("Video A")
        ph_a = st.empty()
    
    with col3:
        st.subheader("Video B")
        ph_b = st.empty()

    ph_a.empty()
    ph_source.empty()
    ph_b.empty()
    
    render_video_iframe(path_a, ph_a)
    render_video_iframe(current_pair['source_path'], ph_source)
    render_video_iframe(path_b, ph_b)

    # 7. Voting Form
    with st.form("voting_form"):        
        st.markdown("#### Imagine you used an AI model to edit the source video based on the prompt. You received these two results, Video A and Video B. Which one are you more likely to accept? \n (Key Considerations: Video Quality, Prompt Alignment, and Motion Fidelity to source video)")
        
        left, middle, right = st.columns([1, 2, 1])

        with right:
            choice = st.radio("", ["Video A", "Video B", "Draw"], index=None, horizontal=True, key=current_session_id, label_visibility="collapsed")
            submitted = st.form_submit_button("Submit", type="primary")
            
        if submitted:
            if choice is None:
                st.error("Please select an option.")
            else:
                if choice == "Draw": winner = "draw"
                elif choice == "Video A": winner = name_a
                else: winner = name_b
                
                file_name = f"testID{st.session_state.test_id}_" + st.session_state.user_session_id + ".csv"
                save_result(current_pair['caption'], current_pair['baseline_name'], winner, file_name)
                st.toast(f"Saved! Winner: {winner}")
                
                st.session_state.task_index += 1
                st.rerun()

    # Progress
    progress_text = f"Evaluation {st.session_state.task_index + 1} of {len(all_pairs)}"
    st.progress((st.session_state.task_index + 1) / len(all_pairs))
    st.caption(f"{progress_text} | Comparison: {current_pair['baseline_name'].upper()} vs OURS")

if __name__ == "__main__":
    main()