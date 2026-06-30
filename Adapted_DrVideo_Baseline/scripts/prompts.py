# prompts.py — Judge/Find loop + open-ended reasoning prompt (JSON outputs only)

from string import Template


def identity(res):
    return res

def first_char_as_answer(res):
    mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    if res and res[0] in mapping:
        return mapping[res[0]]
    return -1

def first_char_after_anchor(anchor):
    def f(res):
        mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        anchor_index = res.find(anchor)
        pred = -1
        if anchor_index >= 0 and anchor_index + len(anchor) < len(res):
            ch = res[anchor_index+len(anchor)]
            if ch in mapping:
                pred = mapping[ch]
        return pred
    return f

def get_intervals_as_list(text):
    text = text.split('.')[0].strip()
    if not text:
        return []
    if text[-1] != ']':
        index = text.rfind(']')
        assert index > 0
        text = text[:index+1]
    interval_list_text = text.split('and')
    intervals = []
    for interval_text in interval_list_text:
        if ',' not in interval_text:
            intervals.append([0, 0])
            continue
        start_text, end_text = interval_text.split(',')
        start_text, end_text = start_text.strip(' []'), end_text.strip(' []')
        if start_text == 'None':
            start_text = '0'
        if end_text == 'None':
            end_text = '1'
        start, end = int(start_text), int(end_text)
        intervals.append([start, end])
    return intervals


class PromptTemplate(object):
    def __init__(self, head, template, post_process_fn):
        self.head = head
        self.prompt_template = template
        self.post_process_fn = post_process_fn

    def get_num_stages(self):
        return len(self.template)

    def get_template_str(self):
        template = []
        for temp in self.prompt_template:
            template.append(temp.safe_substitute())
        return template

    def fill(self, **kwargs):
        prompt_filled = []
        for temp in self.prompt_template:
            prompt_filled.append(temp.substitute(kwargs))
        return prompt_filled


class PromptFactory(object):
    def __init__(self):
        self.prompt_templates = self.build()
    
    def build(self):
        prompt_templates = {}

        # --------- Judge ----------
        prompt_templates['judge'] = PromptTemplate(
            head = "You are a helpful expert in video understanding",
            template = [
                Template("""You are given textual descriptions of a (possibly long) video and a question. 
Each line is 'frame t: ...'. The text may include question-specific augmentations.

Descriptions:
'''
${captions}
'''

Question:
'''
${question_context}
'''

${gpt_prompt}

Task: Decide if the descriptions are sufficient to answer the question accurately and unambiguously.
If YES, return:
{'confidence': '1', 'explanation': ["why the context suffices"]}

If NO, return:
{'confidence': '0', 'explanation': ["what is missing and where to look"]}

Return exactly one of the JSON objects above. No extra text.
""")
            ],
            post_process_fn = identity
        )

        # --------- Find ----------
        prompt_templates['find'] = PromptTemplate(
            head = "You are a helpful expert in video understanding",
            template = [
                Template("""You are given the current descriptions of a long video and a question:
'''
${captions}
'''
Question:
'''
${question_context}
'''
The evidence is insufficient (see explanation):
${explanation}

Select up to 3 new frames to fetch and choose type for each:
A = generic image caption (what is shown?)
B = directed VQA (answer the given question about that frame)

Already covered: A-type at ${type_A}; B-type at ${type_B}.
Return ONLY a JSON list:
[{'frame': 't', 'type': 'A' or 'B'}, ...]
""")
            ],
            post_process_fn = identity
        )

        # --------- Open-ended Reasoning ----------
        prompt_templates['open_reasoning'] = PromptTemplate(
            head = "You are a precise assistant for open-ended video QA.",
            template = [
                Template("""Answer an open-ended question about a long video using only the textual frame descriptions below (each line starts with 'frame t: ...').

Descriptions:
'''
${context}
'''

Question:
'''
${question_text}
'''

Return exactly one JSON object:
{'final_answer': '<short textual answer>', 'rationale': '<1-2 sentence reasoning>'}
""")
            ],
            post_process_fn = identity
        )

        return prompt_templates

    def get(self, prompt_type):
        return self.prompt_templates[prompt_type]
