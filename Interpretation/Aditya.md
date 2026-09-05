# Simple Explanation of the Project

## What is this project about?

This project is about building an AI system that can predict cyber attacks before they become serious.

Instead of only saying whether one network connection is safe or dangerous, the system watches network activity over time. It learns how an attacker usually moves through a network and tries to predict the next step.

For example, it may notice that someone is scanning many ports, then trying to enter a system, and later moving to another computer. The system should warn us about this progress early.

## What information will the system use?

The system will study two types of network information:

- **Flow-level information:** General details about a connection, such as source and destination IP addresses, ports, protocol, number of packets, number of bytes, connection time, and TCP flags.
- **Packet-level information:** More detailed information from individual packets, such as packet size, timing, TTL values, TCP window size, fragments, retransmissions, and signs of port scanning.

Using both types is important. Flow information gives the big picture, while packet information gives small details that may reveal attacks trying to avoid simple detection.

## How will the AI work?

The AI will create a picture of the current network condition. It will then learn how this condition changes from one moment to the next.

The model should:

1. Understand the current network activity.
2. Learn how normal activity and attack activity change over time.
3. Predict what the network may look like in the next few time periods.
4. Estimate the chance that an attack is developing.
5. Work even when it sees an attack pattern that was not exactly present in its training data.

Possible model types include LSTM networks, Transformers, Graph Neural Networks, or other models that are good at understanding sequences and relationships.

## What should the system show?

After receiving a current network traffic file, the system should predict several steps into the future and show:

- The probability that an attack or infiltration will happen soon.
- The likely stage of the attack, such as reconnaissance, initial access, lateral movement, command and control, or data theft.
- The network details that caused the warning, such as unusual ports, TCP flags, packet timing, or other traffic patterns.

The explanation is important. The user should understand why the AI gave a warning instead of seeing only a black-box result.

## What parts need to be built?

The final prototype should include:

- A data processing pipeline that reads CSV network-flow files or PCAP packet-capture files.
- A trained AI world model, including the training code, model files, and settings needed to repeat the training.
- A prediction engine that forecasts the next few network states.
- An explanation feature that describes the main reasons for each prediction in normal language.
- An offline interface, such as a Streamlit app, Flask app, or command-line tool.
- A way to upload a CSV or PCAP file and view the timeline, suspicious connections, and predicted attack stages.

The whole system should be open source and able to run offline.

## How will success be measured?

The project should compare the new AI system with a basic logistic regression model using the same data.

The results should include measures such as:

- F1 score
- Precision
- Recall
- False positive rate

The new system should show a clear and measurable improvement over the basic model.

## What must be submitted?

The project submission should contain:

- The complete source code.
- A README with local installation and setup instructions.
- An architecture document of no more than two pages.
- A demo video of no more than two minutes.
- A technical presentation of no more than five slides explaining the method and results.
