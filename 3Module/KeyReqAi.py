import json
from transformers import pipeline

# Load knowledge base
with open('troubleshooting_knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

# Initialize a simple NLP model
nlp = pipeline('question-answering')

# Get user input
user_input = input("Please describe your problem: ")

# Search knowledge base for a simple text-based match
for issue, details in knowledge_base.items():
    if details["symptom"].lower() in user_input.lower():
        print(f"Possible solution: {details['solution']}")
        break
else:
    print("No matching issue found in the knowledge base.")

def diagnose_network_issue():
    print("Have you restarted your router?")
    response = input("Yes/No: ").strip().lower()
    if response == "no":
        print("Please restart your router and check again.")
    else:
        print("Try resetting your network settings or contacting your provider.")

# Trigger diagnostic logic if the issue is related to the network
if "internet" in user_input.lower():
    diagnose_network_issue()

def automate_fix(issue):
    if issue == "slow_internet":
        print("Resetting network settings...")
        # Simulated network reset
        print("Network settings have been reset. Please check your connection.")
    else:
        print("Automation is not available for this issue.")

# Simulate automatic fix
if "internet" in user_input.lower():
    automate_fix("slow_internet")

def collect_feedback():
    feedback = input("Did this solution resolve your issue? (Yes/No): ").strip().lower()
    if feedback == "yes":
        print("Great! Your feedback has been recorded.")
    else:
        print("We're sorry the issue persists. We'll improve our solution based on your input.")

# Collect feedback after providing a solution
collect_feedback()