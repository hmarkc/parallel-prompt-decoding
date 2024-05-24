# adapted from https://github.com/SafeAILab/EAGLE/blob/d08fe3f23e5f1d986bb50f786af60e5c4f7f757e/eagle/application/webui.py#L4
import os
import time

import gradio as gr
import argparse
from prompt.utils import *
from prompt.model.model import PromptDecoder, AutoPromptDecoder, PromptConfig
from prompt.model.kv_cache import *
import torch
from fastchat.model import get_conversation_template
import re


def truncate_list(lst, num):
    if num not in lst:
        return lst


    first_index = lst.index(num)


    return lst[:first_index + 1]


def find_list_markers(text):

    pattern = re.compile(r'(?m)(^\d+\.\s|\n)')
    matches = pattern.finditer(text)


    return [(match.start(), match.end()) for match in matches]


def checkin(pointer,start,marker):
    for b,e in marker:
        if b<=pointer<e:
            return True
        if b<=start<e:
            return True
    return False


def highlight_text(text, text_list,color="black"):

    pointer = 0
    result = ""
    markers=find_list_markers(text)


    for sub_text in text_list:

        start = text.find(sub_text, pointer)
        if start==-1:
            continue
        end = start + len(sub_text)


        if checkin(pointer,start,markers):
            result += text[pointer:start]
        else:
            result += f"<span style='color: {color};'>{text[pointer:start]}</span>"

        result += sub_text

        pointer = end

    if pointer < len(text):
        result += f"<span style='color: {color};'>{text[pointer:]}</span>"

    return result


def warmup(model):
    conv = get_conversation_template('vicuna')
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    for output_ids in model.ppd_generate(input_ids):
        ol=output_ids.shape[1]


def bot(history, temperature, use_ppd, highlight_ppd,session_state,):
    if not history:
        return history, "0.00 tokens/s", "0.00", session_state
    pure_history = session_state.get("pure_history", [])
    conv = get_conversation_template('vicuna')

    for query, response in pure_history:
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], response)

    prompt = conv.get_prompt()

    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    input_len = input_ids.shape[1]
    naive_text = []
    cu_len = input_len
    totaltime=0
    start_time=time.time()
    total_ids=0
    if use_ppd:

        for output_ids in model.ppd_generate(input_ids, temperature=temperature, max_steps=args.max_new_token):
            totaltime+=(time.time()-start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )
            naive_text.append(model.tokenizer.decode(output_ids[0, cu_len], skip_special_tokens=True,
                                                     spaces_between_special_tokens=False,
                                                     clean_up_tokenization_spaces=True, ))

            cu_len = output_ids.shape[1]
            colored_text = highlight_text(text, naive_text, "orange")
            if highlight_ppd:
                history[-1][1] = colored_text
            else:
                history[-1][1] = text
            pure_history[-1][1] = text
            session_state["pure_history"] = pure_history
            new_tokens = cu_len-input_len
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens/total_ids:.2f}",session_state
            start_time = time.time()


    else:
        for output_ids in model.naive_generate(input_ids, temperature=temperature, max_steps=args.max_new_token):
            totaltime += (time.time() - start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )
            naive_text.append(model.tokenizer.decode(output_ids[0, cu_len], skip_special_tokens=True,
                                                     spaces_between_special_tokens=False,
                                                     clean_up_tokenization_spaces=True, ))
            cu_len = output_ids.shape[1]
            colored_text = highlight_text(text, naive_text, "orange")
            if highlight_ppd and use_ppd:
                history[-1][1] = colored_text
            else:
                history[-1][1] = text
            history[-1][1] = text
            pure_history[-1][1] = text
            new_tokens = cu_len - input_len
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens/total_ids:.2f}",session_state
            start_time = time.time()
            

def user(user_message, history,session_state):
    if history==None:
        history=[]
    pure_history = session_state.get("pure_history", [])
    pure_history += [[user_message, None]]
    session_state["pure_history"] = pure_history
    return "", history + [[user_message, None]],session_state


def regenerate(history,session_state):
    if not history:
        return history, None,"0.00 tokens/s","0.00",session_state
    pure_history = session_state.get("pure_history", [])
    pure_history[-1][-1] = None
    session_state["pure_history"]=pure_history
    if len(history) > 1:  # Check if there's more than one entry in history (i.e., at least one bot response)
        new_history = history[:-1]  # Remove the last bot response
        last_user_message = history[-1][0]  # Get the last user message
        return new_history + [[last_user_message, None]], None,"0.00 tokens/s","0.00",session_state
    history[-1][1] = None
    return history, None,"0.00 tokens/s","0.00",session_state


def clear(history,session_state):
    pure_history = session_state.get("pure_history", [])
    pure_history = []
    session_state["pure_history"] = pure_history
    return [],"0.00 tokens/s","0.00",session_state


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ppd-path",
    type=str,
    default="hmarkc/ppd-vicuna-7b-v1.3",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument(
    "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
)
parser.add_argument(
    "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
)
parser.add_argument(
    "--max-new-token",
    type=int,
    default=512,
    help="The maximum number of new generated tokens.",
)
args = parser.parse_args()

model = AutoPromptDecoder.from_pretrained(
    args.ppd_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
model.cuda()
model.eval()
warmup(model)

custom_css = """
#speed textarea {
    color: red;   
    font-size: 30px; 
}"""

with gr.Blocks(css=custom_css) as demo:
    gs = gr.State({"pure_history": []})
    gr.Markdown('''## PPD Chatbot''')
    with gr.Row():
        speed_box = gr.Textbox(label="Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")
        compression_box = gr.Textbox(label="Compression Ratio", elem_id="speed", interactive=False, value="0.00")

    chatbot = gr.Chatbot(height=600,show_label=False)


    msg = gr.Textbox(label="Your input")
    with gr.Row():
        send_button = gr.Button("Send")
        stop_button = gr.Button("Stop")
        regenerate_button = gr.Button("Regenerate")
        clear_button = gr.Button("Clear")
    
    with gr.Row():
        with gr.Column():
            use_ppd = gr.Checkbox(label="Use PPD", value=True)
            highlight_ppd = gr.Checkbox(label="Highlight the tokens generated by PPD", value=True)
        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="temperature", value=0.5)
    note=gr.Markdown(show_label=False,value='''The Compression Ratio is defined as the number of generated tokens divided by the number of forward passes in the original LLM. If "Highlight the tokens generated by PPD" is checked, the tokens correctly guessed by PPD 
    will be displayed in orange. Note: Checking this option may cause special formatting rendering issues in a few cases, especially when generating code''')
    enter_event=msg.submit(user, [msg, chatbot,gs], [msg, chatbot,gs], queue=True).then(
        bot, [chatbot, temperature, use_ppd, highlight_ppd,gs], [chatbot,speed_box,compression_box,gs]
    )
    clear_button.click(clear, [chatbot,gs], [chatbot,speed_box,compression_box,gs], queue=True)

    send_event=send_button.click(user, [msg, chatbot,gs], [msg, chatbot,gs],queue=True).then(
        bot, [chatbot, temperature, use_ppd, highlight_ppd,gs], [chatbot,speed_box,compression_box,gs]
    )
    regenerate_event=regenerate_button.click(regenerate, [chatbot,gs], [chatbot, msg,speed_box,compression_box,gs],queue=True).then(
        bot, [chatbot, temperature, use_ppd, highlight_ppd,gs], [chatbot,speed_box,compression_box,gs]
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[send_event,regenerate_event,enter_event])
demo.queue()
demo.launch(share=True)