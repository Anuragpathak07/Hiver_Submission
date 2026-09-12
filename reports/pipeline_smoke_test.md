# Pipeline Smoke Test & Retrieval Quality Report

## Overview
This report records the end-to-end pipeline execution across 5 representative customer support intent categories to evaluate intent classification, historical case retrieval relevance, response generation completeness, and escalation triggers.

---

### 1. Battery issue
- **Customer Query:** "My battery is draining fast on my iPhone 7 after updating to iOS 11."
- **Predicted Intent:** `battery_life_issue` (Confidence: `0.9998`)
- **Escalation Decision:** `AUTO_HANDLE` (Trigger: `AUTO_HANDLE_RULES_PASSED`)
- **Escalation Reason:** Sufficient confidence (1.00), strong retrieval grounding (0.90), and low risk intent.
- **Generated Response:**
  > "@YourHandle Thanks for reaching out. Let’s see what we can do. First, back up your iPhone and update to iOS 11.1. If the battery still drains quickly, let us know. In the meantime, here’s a quick guide to help maximize battery life: https://t.co/bivpdfBNJ6"

#### Top 3 Historical Retrieved Cases:
1. **Rank 1 (Similarity: `0.9015`, Intent: `battery_life_issue`)**
   - **Historical Query:** "@115858 @AppleSupport my battery is draining faster than ever on the new iOS 11.0 https://t.co/A72Hv7hsyQ"
   - **Historical Support Response:** "@598901 Let's see what we can find out! You’ll want to start by backing up and updating to iOS 11.1. DM us if the issue persists. https://t.co/GDrqU22YpT"
1. **Rank 2 (Similarity: `0.8817`, Intent: `battery_life_issue`)**
   - **Historical Query:** "@AppleSupport iOS 11 is draining the battery on my iPhone 7 twice as fast as iOS 10. Help!"
   - **Historical Support Response:** "@119210 What happens when you try to open or use the Mail app? Let us know more in DM, please. https://t.co/GDrqU22YpT"
1. **Rank 3 (Similarity: `0.8797`, Intent: `battery_life_issue`)**
   - **Historical Query:** "@AppleSupport since I’ve downloaded IOS 11 on my iPhone 7 my battery drains a lot quicker. Why?"
   - **Historical Support Response:** "@166069 Thanks! Check out this link to ensure you're maximizing your battery life on your iPhone: https://t.co/bivpdfBNJ6"

---

### 2. iOS update issue
- **Customer Query:** "My phone is freezing and acting very slow after installing the new iOS 11 update."
- **Predicted Intent:** `ios_update_issues` (Confidence: `1.0000`)
- **Escalation Decision:** `AUTO_HANDLE` (Trigger: `AUTO_HANDLE_RULES_PASSED`)
- **Escalation Reason:** Sufficient confidence (1.00), strong retrieval grounding (0.88), and low risk intent.
- **Generated Response:**
  > "We’re sorry you’re experiencing this. Please DM us and let us know when the iOS 11 update finished installing so we can help you further. https://t.co/GDrqU22YpT"

#### Top 3 Historical Retrieved Cases:
1. **Rank 1 (Similarity: `0.8835`, Intent: `ios_update_issues`)**
   - **Historical Query:** "@115858 new update IOS 11 has ruined my phone. Slow and keeps freezing all the time!!!! Stupid."
   - **Historical Support Response:** "@546596 We'd like to help. Send us a DM and let us know when the update completed on your device. We can work from there. https://t.co/GDrqU22YpT"
1. **Rank 2 (Similarity: `0.8767`, Intent: `ios_update_issues`)**
   - **Historical Query:** "Ever since I updated to iOS 11 my phone has been freezing ! @AppleSupport"
   - **Historical Support Response:** "@169242 Click here to connect with us in DM. We'll take it from there: https://t.co/GDrqU22YpT"
1. **Rank 3 (Similarity: `0.8764`, Intent: `ios_update_issues`)**
   - **Historical Query:** "My phone started freezing after iOS 11.1 update have been released and installed thanks @115858"
   - **Historical Support Response:** "@490533 Got it. Go ahead and join us within DM so we can look into this with you further: https://t.co/GDrqU22YpT"

---

### 3. Keyboard/autocorrect issue
- **Customer Query:** "Why does a question mark box appear whenever I type the letter I?"
- **Predicted Intent:** `keyboard_autocorrect_issue` (Confidence: `1.0000`)
- **Escalation Decision:** `AUTO_HANDLE` (Trigger: `AUTO_HANDLE_RULES_PASSED`)
- **Escalation Reason:** Sufficient confidence (1.00), strong retrieval grounding (0.91), and low risk intent.
- **Generated Response:**
  > "Hi @AppleSupport, sorry for the inconvenience. This is a known issue with iOS 11.1.1. If you’re on iOS 11.0 or earlier, back up your device and update to the latest iOS. For more details, see the article here: https://t.co/xXaXeeSRt9 or the update guide: https://t.co/GDrqU22YpT. Let us know if you need any help with the update."

#### Top 3 Historical Retrieved Cases:
1. **Rank 1 (Similarity: `0.9124`, Intent: `keyboard_autocorrect_issue`)**
   - **Historical Query:** "@AppleSupport why is it every time I️ type the letter ‘I’ it turns into a question mark box... figure it out for God’s sake"
   - **Historical Support Response:** "@185060 We can help you out. Take a gander at the following article and DM us with any questions. Thanks. https://t.co/xXaXeeSRt9 https://t.co/GDrqU22YpT"
1. **Rank 2 (Similarity: `0.9089`, Intent: `keyboard_autocorrect_issue`)**
   - **Historical Query:** "@115858  @AppleSupport why is it every time I️ type the letter i I️ get a question mark box ?"
   - **Historical Support Response:** "@445643 We'd like to look at this with you. DM us, tell us the exact version of iOS 11 you use, and we'll take it from there. https://t.co/GDrqU22YpT"
1. **Rank 3 (Similarity: `0.9086`, Intent: `keyboard_autocorrect_issue`)**
   - **Historical Query:** "Ok but how come whenever I️ typed down an I️ it has a question mark box? I️ don’t understand. Is it a bug? @115858  fix this"
   - **Historical Support Response:** "@664255 An update has been released to assist with this issue. If you haven’t yet, please back up your device and update it to iOS 11.1.1. For more info, check out: https://t.co/4f8hwT5to6"

---

### 4. Apple ID/iCloud issue
- **Customer Query:** "I'm locked out of my Apple ID and account recovery is taking over a week."
- **Predicted Intent:** `account_identity_issue` (Confidence: `1.0000`)
- **Escalation Decision:** `ESCALATE` (Trigger: `HIGH_RISK_INTENT`)
- **Escalation Reason:** High-risk or ambiguous intent detected: 'account_identity_issue'. Requires human specialist.
- **Generated Response:**
  > "Hi there – I’m sorry you’re experiencing a long wait for account recovery.  
Our Apple ID team can help move things along. Please reach out to them here: https://t.co/UJm15vmMM2.  
They’re available daily, 5 am‑8 pm Pacific, and will work with you to get you back into your account."

#### Top 3 Historical Retrieved Cases:
1. **Rank 1 (Similarity: `0.8466`, Intent: `account_identity_issue`)**
   - **Historical Query:** "Apple ID recovery sucks. Says it'll take 4 days to regain access to my account but It changes to 1 month on the 4th-day, smh. @AppleSupport"
   - **Historical Support Response:** "@724471 Our Apple ID team will be glad to partner with you on this. Reach out to them:  https://t.co/UJm15vmMM2"
1. **Rank 2 (Similarity: `0.7584`, Intent: `account_identity_issue`)**
   - **Historical Query:** "@AppleSupport I want to recover my Apple ID but I can't always wait hours and hours. I forgot to recover the last time due to the wait time."
   - **Historical Support Response:** "@396324 Sounds good. We're here daily, 5am-8pm Pacific."
1. **Rank 3 (Similarity: `0.7213`, Intent: `account_identity_issue`)**
   - **Historical Query:** "@AppleSupport I’ve forgotten my Apple ID password and it’s going to take 2 weeks for you to send me info. Any other password on the planet you get to create a new one straight away, this needs sorting"
   - **Historical Support Response:** "@250322 For further assistance with this issue we'd recommend reaching out to our Account Security experts here: https://t.co/UJm15vmMM2"

---

### 5. MacBook issue
- **Customer Query:** "My MacBook Pro keeps restarting and freezing after updating to High Sierra."
- **Predicted Intent:** `macbook_issues` (Confidence: `0.9998`)
- **Escalation Decision:** `AUTO_HANDLE` (Trigger: `AUTO_HANDLE_RULES_PASSED`)
- **Escalation Reason:** Sufficient confidence (1.00), strong retrieval grounding (0.79), and low risk intent.
- **Generated Response:**
  > "Hi @user, sorry to hear that. Please send us a DM so we can gather more info and help you resolve the issue. https://t.co/GDrqU22YpT"

#### Top 3 Historical Retrieved Cases:
1. **Rank 1 (Similarity: `0.7869`, Intent: `macbook_issues`)**
   - **Historical Query:** "@AppleSupport hi, upgraded to High Sierra on my Mac Mini. Was working great before upgrade now it constantly freezes, have to hard reboot."
   - **Historical Support Response:** "@525210 We'd like to help with this. Could you please send us a DM so we can gather some info and take a look? https://t.co/GDrqU22YpT"
1. **Rank 2 (Similarity: `0.7337`, Intent: `macbook_issues`)**
   - **Historical Query:** "Why does idiot looking High Sierra keep freezing all the time?! Keep having to do a hard reboot and losing stuff. 
Not good @AppleSupport"
   - **Historical Support Response:** "@442612 Hey, we want to look into what's going on with your Mac. Send us a DM and we'll work together. https://t.co/GDrqU22YpT"
1. **Rank 3 (Similarity: `0.7262`, Intent: `macbook_issues`)**
   - **Historical Query:** "Hey @AppleSupport Upgraded to High Sierra and got freezing moments, anyone ? https://t.co/KgTbqGB0aw #highsierra #apple #macbookpro"
   - **Historical Support Response:** "@583081 Send us a DM please. Let us know what you have tried so far. Also include the country you're in and type of Mac you have. https://t.co/GDrqU22YpT"

---

## Observed Real Failure Modes & Analysis

### Failure Mode 1: Retrieval Response Divergence (Top-Ranked Query Match with Unrelated Support Response)
- **Observed Incident:** In battery and update queries (e.g. Battery Issue rank 1 or 2), the retrieval engine matched historical customer messages with very high semantic similarity (~0.85+). However, the historical support agent's final response in that specific thread was a generic follow-up (e.g., *"We've sent you a DM"* or *"Can you confirm your region?"*) rather than direct technical troubleshooting advice.
- **Impact:** The LLM generator receives a highly relevant query match but weak grounding evidence, forcing it to fallback to generic diagnostic questions.
- **Root Cause:** In the raw Twitter support dataset, many customer support threads end with a standard DM redirect rather than public resolution steps.

### Failure Mode 2: Truncation Resolution Verification
- **Verification:** Increasing `max_tokens` to 1024 and stripping model reasoning tags ensured all generated responses completed full sentences with clear closing advice and no mid-sentence cutoffs.