from models import Email, TaskMetadata

EASY_TASK = {
    "metadata": TaskMetadata(
        task_id="easy",
        complexity="easy",
        description="Classify a single email as spam or not_spam, then finish."
    ),
    "inbox": [
        Email(
            id="email_001",
            sender="admin@lottery-winner.net",
            subject="URGENT: Claim your $500,000 prize",
            body="You have been selected! Send us your bank details to claim."
        )
    ],
    "expected_actions": [
        {"action_type": "classify", "email_id": "email_001", "label": "spam"}
    ]
}

MEDIUM_TASK = {
    "metadata": TaskMetadata(
        task_id="medium",
        complexity="medium",
        description="Classify an email (support vs spam) and draft a polite reply if it's support, then finish."
    ),
    "inbox": [
        Email(
            id="email_002",
            sender="customer@example.com",
            subject="Cannot access my account",
            body="Hi, I reset my password but I still can't log in. Please help."
        )
    ],
    "expected_actions": [
        {"action_type": "classify", "email_id": "email_002", "label": "support"},
        {"action_type": "reply", "email_id": "email_002", "reply_keywords": ["password", "reset", "try", "sorry", "support", "help", "look into", "issue", "trouble"]}
    ]
}

HARD_TASK = {
    "metadata": TaskMetadata(
        task_id="hard",
        complexity="hard",
        description="Process a multi-email inbox. Classify all sequentially, set priority for important ones, reply where needed, then finish."
    ),
    "inbox": [
        Email(
            id="email_003",
            sender="boss@company.com",
            subject="Project deadline moved up",
            body="We need the Q3 report by tomorrow. Please prioritize."
        ),
        Email(
            id="email_004",
            sender="spam@offers.com",
            subject="Cheap meds",
            body="Buy now for 90% off."
        ),
        Email(
            id="email_005",
            sender="user@app.com",
            subject="Bug in latest update",
            body="The app crashes on startup on iOS 16. Needs fixing."
        )
    ],
    "expected_actions": [
        {"action_type": "classify", "email_id": "email_003", "label": ["internal", "urgent", "important", "action_required"]},
        {"action_type": "prioritize", "email_id": "email_003", "priority": "high"},
        {"action_type": "classify", "email_id": "email_004", "label": ["spam", "junk", "promotion"]},
        {"action_type": "classify", "email_id": "email_005", "label": ["support", "bug", "urgent", "feedback"]},
        {"action_type": "prioritize", "email_id": "email_005", "priority": "high"},
        {"action_type": "reply", "email_id": "email_005", "reply_keywords": ["fix", "bug", "ios", "update", "patch", "working", "resolve"]},
        {"action_type": "reply", "email_id": "email_003", "reply_keywords": ["report", "q3", "tomorrow", "priority", "working", "send", "done"]}
    ]
}

TASKS = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK
}
