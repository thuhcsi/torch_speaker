#include <algorithm>
#include <numeric>
#include <complex>
#include <vector>
#include <map>
#include <math.h>

class Fbank {
private:
	const double PI = 4*atan(1.0);   // Pi = 3.14...

private:
	// Hertz to Mel conversion
	inline double hz2mel (double f) {
		return 2595*std::log10 (1+f/700);
	}

	// Mel to Hertz conversion
	inline double mel2hz (double m) {
		return 700*(std::pow(10,m/2595)-1);
	}

	void initFilterbank () {

	}

	void PreEmphasis(float coeff, std::vector<float>* data) {
		if (coeff == 0.0) return;
		for (int i = data->size() - 1; i > 0; i--) {
			(*data)[i] -= coeff * (*data)[i - 1];
		}
		(*data)[0] -= coeff * (*data)[0];
	}

public:
	Fbank(int sample_rate=16000){
	}
}

