from manimdef import DefaultManimClass, NotoSerifText
from manim import *


class NameMerger(DefaultManimClass):
    def __init__(self, name1, name2):
        super().__init__()
        self.name1 = name1
        self.name2 = name2

    def construct(self):
        two_name = [self.name1, self.name2]
        name1_str, name2_str = two_name if len(two_name[0]) == len(two_name[1]) else (max(two_name, key=len), min(two_name, key=len))
        merged_name_list, indices1, indices2 = self.merge_name(name1_str, name2_str)
        merged_name = VGroup(*[NotoSerifText(c, color=BLUE_C if i in indices1 else WHITE) for i, c in enumerate(merged_name_list)])\
            .arrange(RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*4)
        if len(merged_name) > 6: self.camera.frame.scale(1.4).shift(DOWN)
        self.play(LaggedStart(
            *[Write(merged_name[i]) for i in indices1], lag_ratio=0.3
            ))
        self.playw(LaggedStart(
            *[Write(merged_name[i]) for i in indices2], lag_ratio=0.3
            ))
        self.playw(merged_name.animate.shift(UP).shift(UP))

        name1_score_list = self.name_to_scores(name1_str)
        name2_score_list = self.name_to_scores(name2_str)
        score1 = VGroup(*[Text(str(c), color=BLUE_C, font="Noto Serif KR", font_size=36).next_to(merged_name[i*2], UP)\
                          for i, c in enumerate(name1_score_list)])
        score2 = VGroup(*[Text(str(c), color=WHITE, font="Noto Serif KR", font_size=36).next_to(merged_name[i*2+1], UP)\
                          for i, c in enumerate(name2_score_list)])
        self.playw(FadeIn(score1, shift=UP))
        self.playw(FadeIn(score2, shift=UP))
        merged_score = [0] * len(merged_name)
        for i, idx in enumerate(indices1): merged_score[idx] = name1_score_list[i]
        for i, idx in enumerate(indices2): merged_score[idx] = name2_score_list[i]
        upper_level = merged_name
        upper_score = merged_score
        n_color = BLUE
        for i in range(len(merged_name), 2, -1):
            new_merged_score = [(upper_score[j]+upper_score[j+1])%10 for j in range(i-1)]
            new_score = []
            for j in range(i-1):
                left, right = upper_level[j].copy(), upper_level[j+1].copy()
                n = Text(str(new_merged_score[j]), color=n_color, font="Noto Serif KR", font_size=36)\
                    .move_to(VGroup(left, right).get_center()).shift(DOWN)
                l1, l2 = DashedLine(left, n, color=GREY, buff=0.1), DashedLine(right, n, color=GREY, buff=0.1)
                self.play(FadeOut(left, target_position=n, scale=0.8), FadeOut(right, target_position=n, scale=0.8),
                          GrowFromCenter(n), Write(l1), Write(l2))
                new_score.append(n)
            n_color = BLUE if n_color == WHITE else WHITE
            upper_level = new_score
            upper_score = new_merged_score
        self.wait()
        #self.playw(Write(VGroup(*new_score).copy().set_color(BLUE_A)), VGroup(*[m for m in self.mobjects[:-2] if isinstance(m, VMobject)]).animate.set_opacity(0.5))
        self.playw(new_score[0].animate.scale(1.5), new_score[1].animate.scale(1.5), VGroup(*[m for i, m in enumerate(self.mobjects) if isinstance(m, VMobject) and i != len(self.mobjects)-4 and i != len(self.mobjects)-7]).animate.set_opacity(0.3))

    def decompose_hangul(self, c):
        c_number = ord(c) - ord("가")
        
        len_jung, len_jong = len(_jung), len(_jong)
        jong_num = c_number % len_jong
        jung_num = (c_number // len_jong) % len_jung
        cho_num  = (c_number // len_jong) // len_jung

        return "".join([_cho[cho_num], _jung[jung_num], _jong[jong_num]])
    
    def merge_name(self, name1, name2):
        merged = []
        name1_list, name2_list = list(name1), list(name2)
        idx, indices1, indices2 = -1, [], []
        while name1_list and name2_list:
            merged.append(name1_list.pop(0))
            indices1.append(idx:=idx+1)
            merged.append(name2_list.pop(0))
            indices2.append(idx:=idx+1)
        if name1_list:
            for _ in name1_list:
                indices1.append(idx:=idx+1)
            merged.extend(name1_list)
        if name2_list:
            for _ in name2_list:
                indices2.append(idx:=idx+1)
            merged.extend(name2_list)

        return merged, indices1, indices2
    
    def name_to_scores(self, name):
        scores = []
        for c in name:
            decomposed = self.decompose_hangul(c)
            cho, jung, jong = decomposed
            score = sum([_cho_hwek[cho], _jung_hwek[jung], _jong_hwek[jong]])
            scores.append(score)
        return scores
    

_cho  = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_jong = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

_cho_hwek = {
    "ㄱ": 1, "ㄲ": 2,
    "ㄴ": 1,
    "ㄷ": 2, "ㄸ": 4,
    "ㄹ": 3,
    "ㅁ": 3,
    "ㅂ": 4, "ㅃ": 8,
    "ㅅ": 2, "ㅆ": 4,
    "ㅇ": 1,
    "ㅈ": 2, "ㅉ": 4,
    "ㅊ": 3,
    "ㅋ": 2,
    "ㅌ": 3,
    "ㅍ": 4,
    "ㅎ": 3
}

_jung_hwek = {
    "ㅏ": 2, "ㅐ": 3, "ㅑ": 3, "ㅒ": 4,
    "ㅓ": 2, "ㅔ": 3, "ㅕ": 3, "ㅖ": 4,
    "ㅗ": 2, "ㅘ": 4, "ㅙ": 5, "ㅚ": 3,
    "ㅛ": 3,
    "ㅜ": 2, "ㅝ": 4, "ㅞ": 5, "ㅟ": 3,
    "ㅠ": 3, "ㅡ": 1, "ㅢ": 2, "ㅣ": 1
}

_jong_hwek = {
    " ": 0,
    "ㄱ": 1, "ㄲ": 2, "ㄳ": 3,
    "ㄴ": 1, "ㄵ": 3, "ㄶ": 4,
    "ㄷ": 2,
    "ㄹ": 3, "ㄺ": 4, "ㄻ": 6, "ㄼ": 7, "ㄽ": 5, "ㄾ": 6, "ㄿ": 7, "ㅀ": 6,
    "ㅁ": 3,
    "ㅂ": 4, "ㅄ": 6,
    "ㅅ": 2, "ㅆ": 4,
    "ㅇ": 1,
    "ㅈ": 2,
    "ㅊ": 3,
    "ㅋ": 2,
    "ㅌ": 3,
    "ㅍ": 4,
    "ㅎ": 3
}