package fb50

import (
	"testing"
)

/*func TestWordBreak(t *testing.T) {
	words := []string{"aaa", "aaaa"}
	s := "aaaaaaa"
	if !WordBreak(s, words) {
		t.Errorf("Test failed for input %v \n", words)
	}
	words = []string{"a", "b"}
	s = "ab"
	if !WordBreak(s, words) {
		t.Errorf("Test failed for input %v \n", words)
	}
}*/

func TestFindKthSmallest(t *testing.T) {
	nums := []int{5, 4, 9, 12, 7, 3}
	k := 1
	output := FindKthSmallest(nums, k)
	if output != 3 {
		t.Errorf("Test failed for input %v. Expected %d but got %d \n", nums, 3, output)
	}
}

func TestBFS(t *testing.T) {
	G := map[int][]int{
		0: []int{1, 2},
		1: []int{0, 2, 3},
		2: []int{0, 1},
		3: []int{1},
	}
	source, target := 0, 3
	pred := BFS(G, source)
	var distance int
	for target != source {
		if parent, ok := pred[target]; ok {
			target = parent
			distance++
		} else {
			break
		}
	}
	if distance != 2 {
		t.Errorf("Distance between node 0 and 3 is incorrect. Expected %d Got %d\n", 2, distance)
	}

}
