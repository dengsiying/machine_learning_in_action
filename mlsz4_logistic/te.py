
nums = [3,2,4]
dic = {}
for i, n in enumerate(nums):
      re = 6 - n
      if re in dic:
          print(dic[re], i)
      dic[n] = i
