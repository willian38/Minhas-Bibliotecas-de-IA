const ActivationFunctions = {
	"linear" : function(x){
		return x;
	},
	"step" : function(x){
		return x >= 0 ? 1 : 0;
	},
	"sigmoid" : function(x){
		return 1 / (1 + Math.exp(-x));
	},
	"tanh" : function(x){
		return 2 * ActivationFunctions["sigmoid"](x) * (2 * ActivationFunctions["sigmoid"](x)) - 1;
	},
	"softmax": function(x,t) {
		var s = 0;
		var m = 0;
		for(var i of t) {
			if(i > m) m = i;
		}
		for(var i of t) {
			s += Math.exp(i - m);
		}
		return Math.exp(x - m) / s;
	},
	"relu" : function(x){
		return Math.max(0,x);
	},
	"leakyRelu" : function(x){
		var a = 0.03;
		return x >= 0 ? x : a*x;
	}
}

const DerivativeFunctions = {
	"linear" : function(x){
		return 1;
	},
	"step" : function(x){
		return x >= 0 ? 1 : 0;
	},
	"sigmoid" : function(x){
		return x * (1 - x);
	},
	"tanh" : function(x){
		return 1 - x**2;
	},
	"softmax": function(x,t) {
		var softmax = ActivationFunctions["softmax"](x, t);
		return softmax * (1 - softmax);
	},
	"relu" : function(x){
		return x >= 0 ? 1 : 0;
	},
	"leakyRelu" : function(x){
		var a = 0.03;
		return x >= 0 ? 1 : a;
	}
}

const LossFunctions = {
	"Default" : function(outputs, target) {
		var delta = [];
		for(var i in outputs) {
			delta[i] = target[i] - outputs[i];
		}
		return delta;
	},
	"MAE" : function(outputs, target) {
		var delta = [];
		for(var i in outputs) {
			delta[i] = 1/2 * (target[i] - outputs[i]);
		}
		return delta;
	},
	"MSE" : function(outputs, target) {
		var delta = [];
		for(var i in outputs) {
			delta[i] = 2 * (target[i] - outputs[i]);
		}
		return delta;
	},
	"CrossEntropy" : function(outputs, target) {
		var delta = [];
		for(var i in outputs) {
			delta[i] = -(target[i] * Math.log(outputs[i]));
			
		}
		return delta;
	}
}

const Matrix = {
	Transpose: function (matrix) {
		let t_matrix = [];

		for(let i = 0; i < matrix.length;i++) {
			for(let j = 0; j < matrix[i].length; j++) {
				if(!t_matrix[j]) t_matrix[j] = [];
				t_matrix[j][i] = matrix[i][j];
			}
		}
		return t_matrix;
	},

	CreateRandom: function (length) {
		let arr = [];
		for(var i = 0; i < length; i++) {
			arr.push(-1 + Math.random() * 2);
		}
		return arr;
	},

	Mul: function(arr1, arr2, add = false) {
		let arr = [];
		let sum = 0;
		for(var i in arr1) {
			sum += arr1[i] * arr2[i];
			arr[i] = arr1[i] * arr2[i];
		}
		if(add) {
			return sum;
		}
		return arr;
	},

	Div: function(arr1, arr2, add = false) {
		let arr = [];
		let sum = 0;
		for(var i in arr1) {
			sum += arr1[i] / arr2[i];
			arr[i] = arr1[i] / arr2[i];
		}
		if(add) {
			return sum;
		}
		return arr;
	},

	Add: function(arr1, arr2, add = false) {
		let arr = [];
		let sum = 0;
		for(var i in arr1) {
			sum += arr1[i] + arr2[i];
			arr[i] = arr1[i] + arr2[i];
		}
		if(add) {
			return sum;
		}
		return arr;
	},

	Sub: function(arr1, arr2, add = false) {
		let arr = [];
		let sum = 0;
		for(var i in arr1) {
			sum += arr1[i] - arr2[i];
			arr[i] = arr1[i] - arr2[i];
		}
		if(add) {
			return sum;
		}
		return arr;
	},

	Softmax: function(arr) {
		let sum = 0;
		let prob = [];
		for(var i of arr) {
			sum += Math.exp(i);
		}
		for(var i in arr) {
			prob[i] = Math.exp(arr[i]) / sum;
		}
		return prob;
	},

	Clipping: function(arr) {
		let newArr = [];
		for(var i in arr) {
			if(arr[i] > 1) {
				newArr[i] = 1;
			}
			else if(arr[i] < -1) {
				newArr[i] = -1;
			}
			else {
				newArr[i] = arr[i];
			}
		}
		return newArr;
	},

	OneShot: function(len,index) {
		let a = [];
		for(var i = 0; i < len; i++) {
			if(index != i) {
				a[i] = 0;
			}
			else {
				a[i] = 1;
			}
		}
		return a;
	}
}

function NeuralNetwork(functions, errorFunction,learnRate) {
    this.functions = functions || null;
	this.lossFunction = errorFunction || "Default";
    this.learnRate = learnRate || 0.01;
	this.momentun = 0.25;
	this.showErr = false;
    this.weights = [];
    this.outputs = [];
    this.inputs = [];
    this.derivatives = [];
    this.delta = [];
    this.bias = [];
	this.error = 0;
	this.sum = [];
	this.gradientInput = [];
    
	this.clipping = function(x) {
        if(x > 1) {
            return 1;
        }
        else if(x < -1) {
            return -1;
        }
        else {
            return x;
        }
    }

	this.normalize = function(data) {
		var min = 0;
		var max = 0;
		var normalized = [];
		for(var value of data) {
			if(value > max) {
				max = value;
			}
			else if(value < min) {
				min = value;
			}
		}

		for(var value of data) {
			normalized.push((value - min) / (max - min));
		}
		return normalized;
	}

	this.getJSON = function() {
		var json = {
			"bias" : this.bias,
			"weights" : this.weights
		}
		return JSON.stringify(json);
	}

	this.loadJSON = function(data) {
		var json = JSON.parse(data);
		this.bias = json.bias;
		this.weights = json.weights;
	}

    this.Gaussian = function (){
		var u , v, s, t, val1, val2;

		do {
			u = 2 * Math.random() - 1;
			v = 2 * Math.random() - 1;
		} while(u * u + v * v > 1 || (u == 0 && v == 0));

		s = u * u + v * v;
		t = Math.sqrt((-2.0 * Math.log(s)) / s);

		val1 = u * t;
		val2 = v * t;

		return val1;
	}

    let layersSize;
    let InputSize;

    this.init = function(layerSize) {

        InputSize = layerSize[0];
        layersSize = layerSize.slice(1,layerSize.length)

        for(var l = 0; l < layersSize.length; l++) {
            this.weights[l] = [];
            this.outputs[l] = [];
            this.derivatives[l] = [];
            this.delta[l] = [];
            this.bias[l] = [];
			this.sum[l] = [];

            for(var i = 0; i < (l == 0 ? InputSize : layersSize[l-1]); i++) {
                this.weights[l][i] = [];

                for(var n = 0; n < layersSize[l];n++) {
                    this.weights[l][i][n] = this.Gaussian();
                }
            }

            for(var n = 0; n < layersSize[l];n++) {
                this.bias[l][n] = this.Gaussian();
                this.outputs[l][n] = 0;
                this.delta[l][n] = 0;
                this.derivatives[l][n] = 0;
            }
        }
    }

	this.TransformToSoftmax = function() {
		for(var i in this.outputs[this.outputs.length-1]) {
			this.outputs[this.outputs.length-1][i] = ActivationFunctions["softmax"](this.outputs[this.outputs.length-1][i],this.outputs[this.outputs.length-1]);
		}
		return this.outputs[this.outputs.length-1];
	}

    this.Run = function(inputs) {
        var output = [];
		this.inputs = inputs;
        for(var l = 0; l < layersSize.length; l++) {
            for(var n = 0; n < layersSize[l];n++) {
                var sum = 0;
                for(var i = 0; i < (l == 0 ? inputs.length : layersSize[l-1]);i++) {
					if(this.weights[l].length !== inputs.length && l == 0) {
						console.error("Entradas e Conexões com tamanhos diferentes! : " + inputs.length + " / " + this.weights[l].length);
						return
					}
                    sum += this.weights[l][i][n] * (l == 0 ? inputs[i] : this.outputs[l-1][i]);
                }
				sum += this.bias[l][n];
                this.sum[l][n] = sum;
            }
			//ativação e derivada da ativação
            for(var n = 0; n < layersSize[l];n++) {
				this.outputs[l][n] = ActivationFunctions[this.functions[l]](this.sum[l][n], this.sum[l]);
				this.derivatives[l][n] = DerivativeFunctions[this.functions[l]](this.outputs[l][n]);
			}
        }
        return output;
    }
	//pega os Outputs de uma layer especifica
	this.getOutsLayer = function(layer) {
		return this.outputs[layer];
	}
	
    this.TrainCase = function(inputs, desired, makeInputGradient, gradients) {
		this.error = 0;
        for(var l = layersSize.length-1;l >= 0; l--) {
			if(l == layersSize.length-1) { //verifica se a camada é a camada de saida
				if(!gradients) { //Verifica se o modelo vai usar gradientes externos, caso contrario execute
					var delta = LossFunctions[this.lossFunction](this.outputs[l], desired)
					for(var k = 0; k < layersSize[l];k++) {
						this.delta[l][k] = this.clipping(delta[k]);
						this.error += this.delta[l][k];
					}
				}
				else { //usa os gradientes externos
					for(var k = 0; k < layersSize[l];k++) {
						this.delta[l][k] = gradients[k];
					}
				}
			}
			else {//propaga o erro para traz
				for(var i = 0; i < layersSize[l];i++) {
					var sum = 0;
					for(var k = 0; k < layersSize[l+1];k++) {
						sum += this.weights[l+1][i][k] * this.delta[l+1][k];
					}
					sum *= this.derivatives[l][i];
					this.delta[l][i] = sum;
				}
			}
		}
		
		if(makeInputGradient) { //propaga o erro para os Inputs para possiveis conexões de modelos diferentes caso ativo
			for(var i = 0; i < this.inputs.length;i++) {
				var sum = 0;
				for(var k = 0; k < layersSize[0];k++) {
					sum += this.weights[0][i][k] * this.delta[0][k];
				}
				this.gradientInput[i] = sum;
			}
		}
		//Ajuste dos pesos e ajuste do Bias
        for(var l = layersSize.length-1;l >= 0; l--) {
			for(var i = 0; i < (l == 0 ? inputs.length : layersSize[l-1]); i++) {
				for(var j = 0; j < layersSize[l]; j++) {

					var delta = this.learnRate * this.delta[l][j] * (l == 0 ? inputs[i] : this.outputs[l-1][i]);
					this.weights[l][i][j] += delta;
				}
			}
			//Ajuste do Bias
			for(var j = 0; j < layersSize[l]; j++) {
				this.bias[l][j] += this.learnRate * this.delta[l][j];
			}
		}
    }
}


function Transformer() {
    let wQ = [];
    let wK = [];
    let wV = [];
    let bQ = [];
    let bK = [];
    let bV = [];
    let Q = [];
    let K = [];
    let V = [];
    let O = []
    let inputs = [];
    let Scores = [];
    let Attention = [];

    this.init = function(states, len) {
        for(var i = 0; i < states; i++) {
            wQ[i] = Matrix.CreateRandom(len);
            wK[i] = Matrix.CreateRandom(len);
            wV[i] = Matrix.CreateRandom(len);
            bQ[i] = -1 + Math.random() * 2;
            bK[i] = -1 + Math.random() * 2;
            bV[i] = -1 + Math.random() * 2;
        }
    }

    this.QKV = function(arr) {
        inputs = arr;
        for(var i in arr) {
            let input = arr[i];
            Q[i] = [];
            K[i] = [];
            V[i] = [];
            for(var neuron in wQ) {
                let sumQ = 0;
                let sumK = 0;
                let sumV = 0;
                for(var w in wQ[neuron]) {
                    sumQ += input[w] * wQ[neuron][w];
                    sumK += input[w] * wK[neuron][w];
                    sumV += input[w] * wV[neuron][w];
                }
                sumQ += bQ[neuron];
                sumK += bK[neuron];
                sumV += bV[neuron];
                Q[i][neuron] = sumQ;
                K[i][neuron] = sumK;
                V[i][neuron] = sumV;
            }
        }
        return {
            Q, K, V
        }
    }

    this.AttentionScore = function() {
        let lastQ = Q[Q.length-1];
        let scores = [];

        for(var i in K) {
            scores[i] = Matrix.Mul(lastQ, K[i], true) / Math.sqrt(lastQ.length);
        }

        Scores = scores;
        Attention = Matrix.Softmax(scores);
        return {
            Scores, Attention
        };
    }

    this.Output = function() {
        let out = [];
        for(var i in Attention) {
            for(var n in V[i]) {
                if(!out[n]) out[n] = 0;
                out[n] += Attention[i] * V[i][n]
				//if(out[n] > 1) out[n] = 1;
				//if(out[n] < -1) out[n] = -1;
            }
        }
        
        O = out;
        return O;
    }

    this.Train = function(gradient,learnRate) {
        for(var i in inputs) {
            let AttentionGradient = [];
            let QueryGradient = [];
            let KeyGradient = [];
            let ValueGradient = [];

            for(var g in gradient) {
                AttentionGradient[g] = gradient[g] * V[i][g] * Attention[i];
                ValueGradient[g] = Attention[i] * gradient[g];
                KeyGradient[g] = AttentionGradient[g] * Q[Q.length-1][g];
                QueryGradient[g] = AttentionGradient[g] * K[i][g];
            }
            for(var neuron in wQ) {
                for(var w in wQ[neuron]){
                    wQ[neuron][w] += learnRate * QueryGradient[neuron] * inputs[i][w];
                    wK[neuron][w] += learnRate * KeyGradient[neuron] * inputs[i][w];
                    wV[neuron][w] += learnRate * ValueGradient[neuron] * inputs[i][w];
                }
    
                bQ[neuron] += learnRate * QueryGradient[neuron];
                bK[neuron] += learnRate * KeyGradient[neuron];
                bV[neuron] += learnRate * ValueGradient[neuron];
            }
        }
        return {
            wQ, wK, wV, bQ, bK, bV
        }
    }
}