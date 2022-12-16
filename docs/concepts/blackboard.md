!!! note
    Our agent is called PAGI, which stands for "PAGI's Artificial General Intelligence".

The **Blackboard** is the data sandbox that the system uses to store all its information. It is composed of nodes and edges, and is directed, hierarchical, dynamic, and persistent. It stores reference data (the current state of the world), knowledge (the current state of the PAGI's mind), tasks (the current state of the PAGI's plans), skills (the current state of the PAGI's abilities and experience), and the self (the current state of the PAGI's processes).

## Abstract Concepts

### Reference Data

Reference data is the current state of the world. It is the data that the PAGI uses to make decisions, and is stored on the Blackboard as a set of facts. Facts are the basic building blocks of reference data, and they need not be objectively true.

### Knowledge

Knowledge is the current state of the PAGI's mind. Together with reference data, it is the data that the PAGI uses to make decisions. Knowledge is stored on the Blackboard as a set of beliefs and plans. Beliefs are used to represent the PAGI's understanding of the world, itself, other agents, and its own processes. They are subjective statements with associated confidence, which is a measure of the PAGI's certainty that the belief is true. Plans can be abstract or specific, and consist of a goal and instructions.

### Tasks

Tasks are the current state of the PAGI's plans. They are the data that the PAGI uses to decide what to do next, and are stored on the Blackboard as an instantiated plan.

### Skills

Skills are the current state of the PAGI's abilities and experience. They are the tools that the PAGI uses to achieve its goals beyond the scope of its own mind. Skills are either built-in or learned, and are stored on the Blackboard as an interface and potential implementation. The interface consists of a description and a set of inputs and outputs, while the implementation is a set of actions that are executed when the skill is called.

### Self

The self is the current state of the PAGI's processes. It is the data that the PAGI uses to decide how to act, and consists of the PAGI's logs and experiences. Logs are used to record the PAGI's actions and experiences, while experiences are used to record the PAGI's successes and failures to learn from them.

## Technical Implementation

PAGI's Blackboard is stored within a git repository, allowing us to utilize git's version control system to track changes to the Blackboard. Each file is a markdown file (**.md**) using the Obsidian markdown format, which is a superset of the standard markdown format. This allows us to use Obsidian's markdown rendering system to view the Blackboard in a web browser or desktop application, and to embed YAML frontmatter in each file to store metadata. Additionally, git's branching system allows us to create multiple versions of the Blackboard for exploration and experimentation, and its merging system enables us to merge changes from different branches into the main branch manually, or to revert to a previous version of the Blackboard if necessary.
