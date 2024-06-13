import numpy as np 

print("Forwards")
forwards_path = "gpt2_scratch_neuro_tokenizer/human_abstracts/PPL_A_and_B.npy"
forwards_PPL_A_and_B = np.load(forwards_path)
mean_PPL_A = np.mean(forwards_PPL_A_and_B[:, 0])
mean_PPL_B = np.mean(forwards_PPL_A_and_B[:, 1])
std_PPL_A = np.std(forwards_PPL_A_and_B[:, 0])
std_PPL_B = np.std(forwards_PPL_A_and_B[:, 1])
print("Mean PPL A: ", mean_PPL_A, "Mean PPL B: ", mean_PPL_B)
print("Std PPL A: ", std_PPL_A, "Std PPL B: ", std_PPL_B)


print("\nBackwards")
backwards_path = "gpt2_scratch_neuro_tokenizer_backwards/human_abstracts/PPL_A_and_B.npy"
backwards_PPL_A_and_B = np.load(backwards_path)
mean_PPL_A = np.mean(backwards_PPL_A_and_B[:, 0])
mean_PPL_B = np.mean(backwards_PPL_A_and_B[:, 1])
std_PPL_A = np.std(backwards_PPL_A_and_B[:, 0])
std_PPL_B = np.std(backwards_PPL_A_and_B[:, 1])
print("Mean PPL A: ", mean_PPL_A, "Mean PPL B: ", mean_PPL_B)
print("Std PPL A: ", std_PPL_A, "Std PPL B: ", std_PPL_B)
