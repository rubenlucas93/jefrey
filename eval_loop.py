import os
import glob
import json
import sys
import unittest.mock as mock
from main import PersonalLLM

def calculate_wer(reference, hypothesis):
    """Placeholder for Word Error Rate."""
    ref_words = reference.lower().replace(".", "").replace(",", "").split()
    hyp_words = hypothesis.lower().replace(".", "").replace(",", "").split()
    
    # Simple naive overlap (can be improved by the agent later)
    matched = sum(1 for w in hyp_words if w in ref_words)
    total = len(ref_words)
    return matched / total if total > 0 else 0

def llm_judge(app, question, expected, actual):
    """Uses the Brain to judge if the actual answer means the same as expected."""
    prompt = (
        f"You are a strict judge. Determine if 'Answer 2' contains the same core factual information as 'Answer 1'.\n"
        f"Question: {question}\n"
        f"Answer 1 (Expected): {expected}\n"
        f"Answer 2 (Actual): {actual}\n"
        f"Reply ONLY with 'YES' or 'NO'."
    )
    result = app.brain.query(prompt, system_prompt="You are a binary judge.")
    return "YES" in result.upper()

def main():
    eval_dir = "data/eval"
    if not os.path.exists(eval_dir):
        print(f"Error: Directory '{eval_dir}' not found.")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(eval_dir, "*_truth.json")))
    if not json_files:
        print("No evaluation templates found.")
        sys.exit(1)

    app = PersonalLLM(debug=False)
    app.skip_mapping = True # Force biometrics

    total_qa_tests = 0
    passed_qa_tests = 0

    print("🚀 Starting Autonomous Evaluation Loop...\n")

    for json_file in json_files:
        base_name = os.path.basename(json_file).replace("_truth.json", "")
        audio_file = os.path.join(eval_dir, f"{base_name}.wav")
        
        if not os.path.exists(audio_file):
            print(f"⚠️ [SKIP] Audio {audio_file} missing for {base_name}.")
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"\nProcessing {base_name}...")
        
        # 1. Clear Memory and Ingest
        app.memory.client.delete_collection(app.memory.collection.name)
        app.memory.collection = app.memory.client.get_or_create_collection(app.memory.collection.name)
        
        # Suppress input globally just in case biometrics fails and it prompts
        with mock.patch('builtins.input', return_value='Rubén'):
            app.ingest_audio(audio_file)

        # 2. Evaluate Q&A
        for qa in data.get("qa_pairs", []):
            q = qa["question"]
            expected = qa["expected_answer"]
            
            # Monkeypatch the print in ask() so we can capture the return value
            # Since ask() doesn't return, we need to extract it from the prompt print
            # Alternatively, we just temporarily modify the brain query
            actual_ans = app.ask(q)
            
            is_correct = llm_judge(app, q, expected, actual_ans)
            total_qa_tests += 1
            if is_correct:
                passed_qa_tests += 1
                print(f"  ✅ Q: {q} -> Correct")
            else:
                print(f"  ❌ Q: {q}\n     Expected: {expected}\n     Got: {actual_ans}")

    if total_qa_tests == 0:
        print("\n⚠️ No tests were run. Did you record the audio files?")
        sys.exit(1)

    final_accuracy = passed_qa_tests / total_qa_tests
    print(f"\n📊 FINAL Q&A ACCURACY: {final_accuracy:.2%}")
    
    if final_accuracy < 0.95:
        print("❌ GOAL NOT MET. AI needs to iterate.")
        sys.exit(1)
    else:
        print("✅ GOAL MET. Pipeline is optimal.")
        sys.exit(0)
