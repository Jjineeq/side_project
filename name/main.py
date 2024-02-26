import streamlit as st
from os import system
from time import sleep
import os

st.markdown("<h1 style='text-align: center; color: #777777;'>이름 궁합 계산기</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #555555;'> E-mail: commit.im@gmail.com / YouTube: <a href=\"http://youtube.com/@ImcommIT\" style=\"opacity:0.5;\"> ImcommIT </a> </h5>", unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])
with c1:
    input_string1 = st.text_input("이름 1")
with c2:
    input_string2 = st.text_input("이름 2")

file_format = "gif"
temp_file_content = f"""
from manim_name import NameMerger
from manim import config

config.video_dir = "./"
config.output_file = "{input_string1}_{input_string2}"
config.format = "{file_format}"
config.quality = "low_quality"
config.partial_movie_dir = "partial_{input_string1}_{input_string2}"

name_merger = NameMerger("{input_string1}", "{input_string2}")
name_merger.render()
"""
def check_hangul(name):
    if not name: return False
    for c in name:
        if ord("가") <= ord(c) <= ord("힣"):
            continue
        else:
            return False
    return True


def show_mp4(file):
    video_bytes = open(f"{input_string1}_{input_string2}.mp4", "rb").read()
    st.video(video_bytes)

import base64
def show_gif(file):
    with open(file, "rb") as fr:
        contents = fr.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        st.image(file, )
        # st.markdown(
        # f'<img src="data:image/gif;base64,{data_url}" alt="gif" max-width="3%">',
        # unsafe_allow_html=True,
        # )

if 'run_button' in st.session_state and st.session_state.run_button == True:
    st.session_state.running = True
else:
    st.session_state.running = False

c3, _, c4 = st.columns([3, 5, 1])
c5, = st.columns([1])
complete=False
with c3:
    if st.button("Match name!", disabled=st.session_state.running, key='run_button'):
        if not (check_hangul(input_string1) and check_hangul(input_string2)):
            st.warning("한글 이름을 입력해주세요", icon="⚠️")
        else:
            while complete != True:
                try:
                    with open(f"{input_string1}_{input_string2}.py", "w", encoding="utf-8") as fw:
                        fw.write(temp_file_content)
                    system(f"python {input_string1}_{input_string2}.py")
                    sleep(0.1)
                    with c5:
                        if file_format == "mp4":
                            show_mp4(f"{input_string1}_{input_string2}.{file_format}")
                        elif file_format == "gif":
                            show_gif(f"{input_string1}_{input_string2}.{file_format}")
                    os.remove(f"{input_string1}_{input_string2}.{file_format}")  # rm 대신 os.remove 사용
                    os.remove(f"{input_string1}_{input_string2}.py")  # rm 대신 os.remove 사용
                    os.system(f"rmdir /s /q partial_{input_string1}_{input_string2}")  # rm -r 대신 rmdir /s /q 사용
                    with open(f"history/{input_string1}_{input_string2}.txt", "w") as fw:
                        fw.write("0")
                    complete=True
                except:
                    pass


with c4:
    if st.button("Reset!", key="reset"):
        st.session_state.running=False
        complete = False
