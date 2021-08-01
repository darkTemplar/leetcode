package str

import "fmt"

// P3 - Longest substring without repeating chars
func lengthOfLongestSubstring(s string) int {
	used := make(map[byte]int)
	var max, left, right int
	for right < len(s) {
		if _, ok := used[s[right]]; ok {
			delete(used, s[left])
			left++
		} else {
			used[s[right]] = right
			right++
		}
		if (right - left) > max {
			max = right - left
		}
	}
	return max
}

func groupStrings(strings []string) [][]string {
	var res [][]string
	groups := map[string][]string{}
	var code string
	for _, s := range strings {
		code = encodeStr(s)
		if group, ok := groups[code]; ok {
			group = append(group, s)
			groups[code] = group
		} else {
			groups[code] = []string{s}
		}
		fmt.Println(groups)
	}
	for _, v := range groups {
		res = append(res, v)
	}
	return res
}

//P 249 - Group Shifted strings
func encodeStr(s string) string {
	r := []rune(s)
	var diff int
	var res string
	for i := 1; i < len(r); i++ {
		diff = int(r[i] - r[i-1])
		fmt.Println(diff)
		if diff < 0 {
			diff += 26
		}
		res += "a" + string(diff) + ","
	}

	return res
}
