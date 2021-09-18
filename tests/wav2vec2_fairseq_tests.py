import os
import sys
import unittest

sys.path.insert(0, os.path.dirname("../bol"))

from bol.models import load_model

# class SingleFile(unittest.TestCase):
#     def setUp(self):
#         self.single_kenlm_output = 'आठ बजे के बीच वनप्लस की ऑफिशियल वेबसाइट पर होगी हालांकि इसके लिए यूजर्स को पहले ही रजिस्टर करना होगा रजिस्टर करने के लिए यूजर्स को वन प्लस अकाउंट बनाना'
#         self.single_viterbi_output = 'आठ बजे के बीच वन प्लस की ऑफिशियल वेबसाइट पर होगी हालांकि इसके लिए यूज्स को पहले ही रजिस्टर कराना होगा रजिस्टर करााने के लिए यूजर्स को वन प्लस अकाउंट बनाना'

#     def test_load_model_with_default_parameters(self):
#         model = load_model('hi-IN')
#         pred = model.predict(['../demos/test_audios/single/ahd_28_long_1335_hin-002500-005500-1-1.wav'])
#         self.assertEqual( pred[0]['transcription'], self.single_viterbi_output, "Should be same")

#     def test_load_model_with_default_parameters_lm(self):
#         model = load_model('hi-IN')
#         pred = model.predict(['../demos/test_audios/single/ahd_28_long_1335_hin-002500-005500-1-1.wav'], with_lm = True)
#         self.assertEqual( pred[0]['transcription'], self.single_kenlm_output, "Should be same")

#     def test_load_model_custom_1(self):
#         model = load_model('hi-IN', use_lm= False)
#         pred = model.predict(['../demos/test_audios/single/ahd_28_long_1335_hin-002500-005500-1-1.wav'])
#         self.assertEqual( pred[0]['transcription'], self.single_viterbi_output, "Should be same")

#     def test_load_model_custom_2(self):
#         model = load_model('hi-IN', use_lm= False)
#         pred = model.predict(['../demos/test_audios/single/ahd_28_long_1335_hin-002500-005500-1-1.wav'], with_lm = True)
#         self.assertEqual( pred[0]['transcription'], self.single_viterbi_output, "Should be same")


class Directory(unittest.TestCase):
    def setUp(self) -> None:
        self.single_kenlm_output = "आठ बजे के बीच वनप्लस की ऑफिशियल वेबसाइट पर होगी हालांकि इसके लिए यूजर्स को पहले ही रजिस्टर करना होगा रजिस्टर करने के लिए यूजर्स को वन प्लस अकाउंट बनाना"
        self.single_viterbi_output = "आठ बजे के बीच वन प्लस की ऑफिशियल वेबसाइट पर होगी हालांकि इसके लिए यूज्स को पहले ही रजिस्टर कराना होगा रजिस्टर करााने के लिए यूजर्स को वन प्लस अकाउंट बनाना"

    # def test_load_model_from_dir_1(self):
    #     model = load_model('hi-IN')
    #     pred = model.predict_from_dir(dir_path='/home/harveen/bol/dev/eval', ext='wav')
    #     local_pred = ''
    #     for item in pred:
    #         if item['file'].split('/')[-1] == 'ahd_28_long_1335_hin-002500-005500-1-1.wav':
    #             local_pred = item['transcription']
    #     self.assertEqual( local_pred, self.single_viterbi_output, "Should be same")

    # def test_load_model_from_dir_2(self):
    #     model = load_model('hi-IN')
    #     pred = model.predict_from_dir(dir_path='/home/harveen/bol/dev/eval', ext='wav', with_lm=True)
    #     local_pred = ''
    #     for item in pred:
    #         if item['file'].split('/')[-1] == 'ahd_28_long_1335_hin-002500-005500-1-1.wav':
    #             local_pred = item['transcription']
    #     self.assertEqual( local_pred, self.single_kenlm_output, "Should be same")

    def test_load_model_from_dir_3(self):
        model = load_model("hi-vakyansh", use_lm=False)
        pred = model.predict_from_dir(dir_path="/home/harveen/bol/dev/eval", ext="wav")
        local_pred = ""
        for item in pred:
            if (
                item["file"].split("/")[-1]
                == "ahd_28_long_1335_hin-002500-005500-1-1.wav"
            ):
                local_pred = item["transcription"]
        self.assertEqual(local_pred, self.single_viterbi_output, "Should be same")

    # def test_load_model_from_dir_4(self):
    #     model = load_model('hi-IN', use_lm=False)
    #     pred = model.predict_from_dir(dir_path='/home/harveen/bol/dev/eval', ext='wav', with_lm=True)
    #     local_pred = ''
    #     for item in pred:
    #         if item['file'].split('/')[-1] == 'ahd_28_long_1335_hin-002500-005500-1-1.wav':
    #             local_pred = item['transcription']
    #     self.assertEqual( local_pred, self.single_viterbi_output, "Should be same")


if __name__ == "__main__":
    unittest.main(verbosity=2)
