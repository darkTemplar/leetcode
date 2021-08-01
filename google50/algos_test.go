package google50

import (
	"testing"
)

/*func TestDecodeString(t *testing.T) {
	in := "3[a]2[b]"
	out := DecodeString(in)
	if out != "aaabb" {
		t.Errorf("Test failed for input 3[a]2[b]. Output %s\n", out)
	}
}*/

func TestMaxScore(t *testing.T) {
	arr := []int{1, 79, 80, 1, 1, 200, 1}
	sum := MaxScore(arr, 3)
	if sum != 202 {
		t.Errorf("Test failed for input %v. Output %d\n", arr, sum)
	}
}

func TestGcd(t *testing.T) {
	a, b := 12, 8
	gcd := Gcd(a, b)
	if gcd != 4 {
		t.Errorf("Test failed for input %d and %d. Output %d\n", a, b, gcd)
	}
}

func TestLIS(t *testing.T) {
	arr := []int{10, 9, 2, 5, 3, 7, 101, 18}
	lis := LIS(arr)
	if lis != 4 {
		t.Errorf("Test failed for input %v. Output %d instead of %d\n", arr, lis, 4)
	}
	arr = []int{1, 3, 6, 7, 9, 4, 10, 5, 6}
	lis = LIS(arr)
	if lis != 6 {
		t.Errorf("Test failed for input %v. Output %d instead of %d\n", arr, lis, 6)
	}
}
