
def match_words(words):
  ctr = 0
  for word in words:
    if len(word) > 6:
      firstHalf = word[0:3:2]
      secondHalf = word[-3::2]
      if(firstHalf == secondHalf):
        print(word)
        ctr += 1
  return ctr

# print(match_words(['abc', 'xyz', 'aba', '1221']))
print(match_words(['abcdabc', 'aasssdxyz', 'asdasdaba', '123dasd123']))

array = [[ ['*' for col in range(6)] for col in range(4)] for row in range(3)]
print(array)


# Python3 program to find the
# number of charters in the
# longest word in the sentence.
def LongestWordLength(str):
    n = len(str)
    res = 0;
    curr_len = 0

    for i in range(0, n):

        # If current character is
        # not end of current word.
        if (str[i] != ' '):
            curr_len += 1

        # If end of word is found
        else:
            res = max(res, curr_len)
            curr_len = 0

    # We do max one more time to consider
    # last word as there won't be any space
    # after last word.
    return max(res, curr_len)


# Driver Code
s = "I am an intern at geeksforgeeks"
print(LongestWordLength(s))

# This code is contribute by Smitha Dinesh Semwal.

s = ["deepak", "swami", "abrahim", "deepak"]

s.sort()
print(s.count("swami"))
print(s)
