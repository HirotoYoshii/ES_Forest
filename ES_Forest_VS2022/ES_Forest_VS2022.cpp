#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <chrono>
using namespace std;

enum class Program_Type
{
	Simple_Random_Forest = 0, // Simple Random Forest
	Single_ES_Forest = 1,     // Single Exponentiation Space Forest
	Double_ES_Forest = 2,     // Single Exponentiation Space Forest
	Just_Select_Samples = 3,  // Just Select Samples
};
Program_Type g_Program_Type = Program_Type::Single_ES_Forest;

enum class RF_Type
{
	Projection_1 = 0, // Projection to Exponentiation Space 1
	Projection_2 = 1, // Projection to Exponentiation Space 2
	Recognition = 2,  // Recognition
};
RF_Type g_RF_Type = RF_Type::Recognition;

enum class Query_Ptr_Type
{
	Dimension_Ptr = 0, // query based on dimension pointer
	Instance_Ptr = 1,  // query based on instance pointer
};
Query_Ptr_Type g_Query_Ptr_Type = Query_Ptr_Type::Dimension_Ptr;

#pragma region parameters
const int Label_Num = 10; // 0--9 digit
const int _SIZE_ = 28; // 28x28 pixel image size
const int AREA_SIZE = (_SIZE_ * _SIZE_); // 784 pixel area size

const bool File_Output = false; // true: output, false: no output
const bool Do_Training_Projection_1 = true; // true: do training projection 1, false: no training projection 1
const bool Do_Training_Projection_2 = true; // true: do training projection 2, false: no training projection 2
const bool Do_Training_Recognition = true; // true: do training recognition, false: no training recognition

const int Partition_Num = 4; // number of partitions for training data
const int Tree_Num_for_Projection_1 = 64; // number of trees for projection 1
const int Tree_Num_for_Projection_2 = 64; // number of trees for projection 2
const int Tree_Num_for_Recognition = 64; // number of trees for recognition
//const int Tree_Num_for_Recognition = 512; // number of trees for recognition for Simple_Random_Forest

const float Min_Entropy = 0.0001;
float g_Leaf_Entropy = Min_Entropy;

int Query_Candidate_Num = 8;
//int Query_Candidate_Num = 16; // number of query candidates for Simple_Random_Forest
const int Query_Eval_Sample_Num = 1024;

const float Query_Offset_Range = 0.2;
uniform_real_distribution<float> offset_dist(0.0, Query_Offset_Range);
uniform_real_distribution<float> ab_dist(0.0, 1.0);
uniform_real_distribution<float> ab_dist_Pair(0.9, 1.0);
#pragma endregion

#pragma region Global Variables
//----------------------------------------
float* g_Training_Data = nullptr;
const int Dimension_Num = AREA_SIZE;

float* g_Training_Data_1 = nullptr;
const int Dimension_Num_1 = AREA_SIZE * Label_Num;

float* g_Training_Data_2 = nullptr;
const int Dimension_Num_2 = AREA_SIZE * Label_Num * Label_Num;

unsigned char* g_Training_Label = nullptr;
unsigned char* g_Training_Label_1 = nullptr;
unsigned char* g_Training_Label_2 = nullptr;

int g_Sample_Num = 0;
float* g_Input_Data = nullptr;
int g_Input_Dimension_Num = 0;
unsigned char* g_Input_Label = nullptr;

vector<int> g_Ptr_Arrays[Partition_Num];
//----------------------------------------
#pragma endregion

#pragma region memory sequence constant and union
//-------------------------------------------
union union_4byte
{
	float float_value;
	int32_t int_value;
	uint32_t uint_value;
};

union Compact_Leaf_Info
{
	uint16_t count[Label_Num];
	uint32_t uint_array[Label_Num / 2];
};
//-------------------------------------------
#pragma endregion

#pragma region Global_Functions
// Set Partition ptr arrays with random sample indices
void Set_g_Ptr_Arrays()
{
	mt19937_64 random_engine(0);
	vector<int> ptr_array;
	for (int n = 0; n < g_Sample_Num; n++) ptr_array.push_back(n);
	shuffle(ptr_array.begin(), ptr_array.end(), random_engine);
	for (int n = 0; n < g_Sample_Num; n++) {
		g_Ptr_Arrays[n % Partition_Num].push_back(ptr_array[n]);
	}
}

float Calc_Norm(const float* a, const float* b, int dimension_num)
{
	float ret = 0.0;

#pragma omp simd
	for (int n = 0; n < dimension_num; n++) {
		ret += (a[n] - b[n]) * (a[n] - b[n]);
	}

	return ret;
}

int Load_Data(const char* file_name, float** data, int& sample_num, const int dimension_num)
{
	FILE* in = nullptr;

	if (fopen_s(&in, file_name, "rb") != 0) return -1;
	fread(&sample_num, sizeof(int), 1, in);

	int temp = 0;
	fread(&temp, sizeof(int), 1, in);
	if (temp != dimension_num) return -1;

	delete[] * data;

	*data = new float[sample_num * dimension_num];
	fread(*data, sizeof(float), sample_num * dimension_num, in);
	fclose(in);

	return 0;
}

int Load_Label(const char* file_name, unsigned char** data, const int sample_num)
{
	FILE* in = nullptr;
	if (fopen_s(&in, file_name, "rb") != 0) return -1;
	int temp = 0;
	fread(&temp, sizeof(int), 1, in);
	if (temp != sample_num) return -1;

	delete[] * data;
	*data = new unsigned char[sample_num];
	fread(*data, sizeof(unsigned char), sample_num, in);
	fclose(in);

	return 0;
}

float Entropy(const int* count_array)
{
	float entropy = 0.0;
	float sum = 0.0;

	for (int n = 0; n < Label_Num; n++) {
		sum += (float)count_array[n];
	}

	if (sum <= 0.0)	return 0.0;

	float prob = 0.0;
	for (int n = 0; n < Label_Num; n++) {
		prob = (float)count_array[n] / sum;
		if (prob > 0.0)
			entropy -= prob * log(prob);
	}

	return entropy;
}

int Count_Label(const vector<int>& ptr_array, int* count_array)
{
	int total_count = 0;

	for (int n = 0; n < Label_Num; n++) {
		count_array[n] = 0;
	}

	for (auto ptr : ptr_array) {
		count_array[g_Input_Label[ptr]] += 1;
		total_count += 1;
	}

	return total_count;
}

float Avg_Entropy(vector<int> ptr_arrays[2])
{
	int count_arrays[2][Label_Num];
	int total_count = 0;
	int each_branch_count[2];

	for (int branch = 0; branch < 2; branch++) {
		each_branch_count[branch] = Count_Label(ptr_arrays[branch], count_arrays[branch]);
		total_count += each_branch_count[branch];
	}

	float ret = 0.0;
	for (int branch = 0; branch < 2; branch++) {
		ret += (float)each_branch_count[branch] * Entropy(count_arrays[branch]);
	}

	return ret / (float)total_count;
}
#pragma endregion

const int QI_Compact_Size = 5;
class Query_Info
{
public:
	int base = 0;
	int look = 0;

	float a = 1.0;
	float b = 1.0;
	float offset = 0.0;

	void set(const union_4byte data[QI_Compact_Size])
	{
		this->base = data[0].int_value;
		this->look = data[1].int_value;

		this->a = (float)(data[2].float_value);
		this->b = (float)(data[3].float_value);
		this->offset = (float)(data[4].float_value);
	}

	void get(union_4byte data[QI_Compact_Size])
	{
		data[0].int_value = this->base;
		data[1].int_value = this->look;

		data[2].float_value = (float)a;
		data[3].float_value = (float)b;
		data[4].float_value = (float)offset;
	}

	// Calculate branch number (0/1) based on value_array
	int Calc_Branch(const float* value_array)
	{
		switch (g_Query_Ptr_Type)
		{
		case Query_Ptr_Type::Dimension_Ptr:
		{
			float base_value = a * value_array[base];
			float look_value = b * value_array[look];

			if ((look_value - base_value + offset) <= 0)
				return 0;
			else
				return 1;
		}

		case Query_Ptr_Type::Instance_Ptr:
		{
			int dim_num = g_Input_Dimension_Num;

			float base_norm = a * Calc_Norm(g_Input_Data + base * dim_num, value_array, dim_num);
			float look_norm = b * Calc_Norm(g_Input_Data + look * dim_num, value_array, dim_num);

			if ((look_norm - base_norm + offset) <= 0)
				return 0;
			else
				return 1;
		}
		}
	}
};

class Node
{
public:
	void Set_Leaf_Sequence(int* count_array, vector<union_4byte>& memory_sequence, const vector<int>& remaining_ptr)
	{
		switch (g_RF_Type)
		{
		case RF_Type::Recognition:
		{
			Compact_Leaf_Info leaf_info;
			for (int n = 0; n < Label_Num; n++) {
				if (count_array[n] >= (1 << 16)) {
					printf("count overflow\n");
					leaf_info.count[n] = (uint16_t)(count_array[n] & 0x0000ffff);
				}
				else {
					leaf_info.count[n] = (uint16_t)(count_array[n]);
				}
			}

			for (int n = 0; n < Label_Num / 2; n++) {
				union_4byte content;
				content.uint_value = leaf_info.uint_array[n];
				memory_sequence.push_back(content);
			}
		}
		break;

		case RF_Type::Projection_1:
		case RF_Type::Projection_2:
		{
			vector<int> label_array;
			for (int n = 0; n < Label_Num; n++) {
				if (count_array[n] > 0) {
					label_array.push_back(n);
				}
			}

			union_4byte header;
			header.int_value = label_array.size();
			memory_sequence.push_back(header);

			float total_count = 0.0;
			for (int n = 0; n < Label_Num; n++) {
				total_count += (float)count_array[n];
			}

			// store label and average pattern of the label
			for (auto label : label_array) {
				union_4byte data;
				data.int_value = label;
				memory_sequence.push_back(data);

				float* pattern = new float[g_Input_Dimension_Num];
				memset(pattern, 0, g_Input_Dimension_Num * sizeof(float));
				for (auto ptr : remaining_ptr) {
					if (g_Input_Label[ptr] == label) {
						for (int i = 0; i < g_Input_Dimension_Num; i++) {
							pattern[i] += g_Input_Data[ptr * g_Input_Dimension_Num + i] / total_count;
						}
					}
				}

				for (int i = 0; i < g_Input_Dimension_Num; i++) {
					data.float_value = (float)pattern[i];
					memory_sequence.push_back(data);
				}
				delete[] pattern;
			}
		}
		break;
		}
	}

	void Make_Children(int depth, int& Leaf_Num, const vector<int>& remaining_ptr, vector<union_4byte>& memory_sequence, mt19937_64& random_engine)
	{
		float* data = g_Input_Data;
		int Label_Count[Label_Num];

		//--------------------------------------
		int header_ptr = memory_sequence.size();
		union_4byte header;
		header.uint_value = 0;
		memory_sequence.push_back(header);
		//--------------------------------------

		Count_Label(remaining_ptr, Label_Count);
		float group_entropy = 0.0;
		float entropy = Entropy(Label_Count);

		if (entropy <= g_Leaf_Entropy) {
			Leaf_Num++;
			Set_Leaf_Sequence(Label_Count, memory_sequence, remaining_ptr);
			return;
		}

		int eval_sample_num = min(Query_Eval_Sample_Num, static_cast<int>(remaining_ptr.size()));

		uniform_int_distribution<int> dist(0, g_Input_Dimension_Num - 1);

		vector<Query_Info> query_candidate;
		int query_candidate_num = Query_Candidate_Num * (depth + 1);

		vector<int> remaining_label_array;
		for (int n = 0; n < Label_Num; n++) if (Label_Count[n] > 0) remaining_label_array.push_back(n);

		for (int n = 0; n < query_candidate_num; n++) {
			Query_Info _query;

			switch (g_Query_Ptr_Type)
			{
			case Query_Ptr_Type::Dimension_Ptr:
				_query.base = dist(random_engine);
				do {
					_query.look = dist(random_engine);
				} while (_query.base == _query.look);

				_query.a = ab_dist(random_engine);
				_query.b = ab_dist(random_engine);
				_query.offset = offset_dist(random_engine);

				break;

			case Query_Ptr_Type::Instance_Ptr:
			{
				//---------------------------------------------------------------------------------
				shuffle(remaining_label_array.begin(), remaining_label_array.end(), random_engine);

				int base_label = remaining_label_array[0];
				int look_label = remaining_label_array[1];

				vector<int> base_ptr_array;
				vector<int> look_ptr_array;

				for (auto ptr : remaining_ptr) {
					if (g_Input_Label[ptr] == base_label) base_ptr_array.push_back(ptr);
					if (g_Input_Label[ptr] == look_label) look_ptr_array.push_back(ptr);
				}

				uniform_int_distribution<int> base_dist(0, base_ptr_array.size() - 1);
				uniform_int_distribution<int> look_dist(0, look_ptr_array.size() - 1);

				_query.base = base_ptr_array[base_dist(random_engine)];
				_query.look = look_ptr_array[look_dist(random_engine)];
				//---------------------------------------------------------------------------------

				_query.a = ab_dist_Pair(random_engine);
				_query.b = ab_dist_Pair(random_engine);
				//_query.offset = offset_dist(random_engine);

				break;
			}
			}

			query_candidate.push_back(_query);
		}

		Query_Info max_query;
		float max_score = -1.0;

		for (auto& _query : query_candidate) {
			vector<int> ptr_arrays[2];
			for (int i = 0; i < eval_sample_num; i++) {
				int ptr = remaining_ptr[i];
				int branch = _query.Calc_Branch(data + ptr * g_Input_Dimension_Num);
				ptr_arrays[branch].push_back(ptr);
			}

			if (ptr_arrays[0].size() * ptr_arrays[1].size() == 0) {
				continue;
			}

			float score = entropy - Avg_Entropy(ptr_arrays);
			if (score > max_score) {
				max_score = score;
				max_query = _query;
			}
		}

		union_4byte compact_query_info[QI_Compact_Size];
		max_query.get(compact_query_info);
		for (int n = 0; n < QI_Compact_Size; n++) {
			memory_sequence.push_back(compact_query_info[n]);
		}

		vector<int> child_ptr_array[2];
		for (auto ptr : remaining_ptr) {
			int branch = max_query.Calc_Branch(data + ptr * g_Input_Dimension_Num);
			child_ptr_array[branch].push_back(ptr);
		}

		for (int branch = 0; branch < 2; branch++) {
			Node child_node;
			child_node.Make_Children(depth + 1, Leaf_Num, child_ptr_array[branch], memory_sequence, random_engine);

			if (branch == 0) {
				union_4byte header;
				header.uint_value = memory_sequence.size();
				memory_sequence[header_ptr] = header;
			}
		}
	}
};

class Tree
{
public:
	vector<union_4byte> memory_sequence;

	int Train(int tree_number, char* file_name)
	{
		Node root;
		this->memory_sequence.clear();

		mt19937_64 random_engine(tree_number);
		vector<int> ptr_array;
		switch (g_RF_Type)
		{
		case RF_Type::Projection_1:
		case RF_Type::Projection_2:
		{
			int partition_n = 0;
			if (g_RF_Type == RF_Type::Projection_1)
				partition_n = tree_number / Tree_Num_for_Projection_1;
			else if (g_RF_Type == RF_Type::Projection_2)
				partition_n = tree_number / Tree_Num_for_Projection_2;

			for (int n = 0; n < Partition_Num; n++) {
				if (n == partition_n) continue;
				for (auto ptr : g_Ptr_Arrays[n]) ptr_array.push_back(ptr);
			}
			break;
		}

		case RF_Type::Recognition:
		{
			for (int i = 0; i < g_Sample_Num; i++) ptr_array.push_back(i);
			break;
		}
		}

		shuffle(ptr_array.begin(), ptr_array.end(), random_engine);

		int Leaf_Num = 0;
		root.Make_Children(0, Leaf_Num, ptr_array, this->memory_sequence, random_engine);

		if (File_Output) {
			FILE* out;
			if (fopen_s(&out, file_name, "wb") == 0) {
				for (auto& value : memory_sequence) {
					fwrite(&value, sizeof(value), 1, out);
				}
				fclose(out);
			}
		}

		return Leaf_Num;
	}

	void Load(int tree_number, char* file_name)
	{
		FILE* in;

		memory_sequence.clear();
		if (fopen_s(&in, file_name, "rb") == 0) {
			union_4byte value;
			while (fread(&value, sizeof(value), 1, in) == 1) {
				memory_sequence.push_back(value);
			}
			fclose(in);
		}
	}

	// Get leaf pointer <<in memory sequence>> based on value_array
	int Get_Leaf_Ptr(const float* value_array)
	{
		union_4byte header;
		int ptr = 0;
		for (;;) {
			header = memory_sequence[ptr++];
			if (header.uint_value == 0) break; // Leaf

			union_4byte query_data[QI_Compact_Size];
			for (int n = 0; n < QI_Compact_Size; n++) {
				query_data[n] = memory_sequence[ptr + n];
			}

			Query_Info query;
			query.set(query_data);
			int branch = query.Calc_Branch(value_array);

			if (branch == 0) {
				ptr = ptr + QI_Compact_Size;
			}
			else { // branch == 1
				ptr = header.uint_value;
			}
		}
		return ptr;
	}
};

class Random_Forest
{
public:
	vector<Tree> tree_array;

	void Initialize(int tree_num)
	{
		tree_array.clear();
		for (int n = 0; n < tree_num; n++) {
			Tree tree;
			tree_array.push_back(tree);
		}
	}

	void fit()
	{
#pragma omp parallel for num_threads(8)
		for (int n = 0; n < tree_array.size(); n++) {
			char file_name[256];
			switch (g_RF_Type)
			{
			case RF_Type::Projection_1:
				sprintf_s(file_name, "tree_projection_1.%05d", n);
				break;

			case RF_Type::Projection_2:
				sprintf_s(file_name, "tree_projection_2.%05d", n);
				break;

			case RF_Type::Recognition:
				sprintf_s(file_name, "tree.%05d", n);
				break;
			}

			int Leaf_Num = tree_array[n].Train(n, file_name);
			printf("tree %d Leaf_Num = %d\n", n, Leaf_Num);
		}
	}

	void load_from_file(const char* name)
	{
		for (int n = 0; n < tree_array.size(); n++) {
			char file_name[256];
			sprintf_s(file_name, "%s.%05d", name, n);
			tree_array[n].Load(n, file_name);
		}
	}

	// count up sum_array based on value_array
	void sum_count(const float* value_array, vector<float>& sum_array)
	{
		int* count_array = new int[tree_array.size() * Label_Num];

#pragma omp parallel for num_threads(8)
		for (int tree_n = 0; tree_n < tree_array.size(); tree_n++) {
			int ptr = tree_array[tree_n].Get_Leaf_Ptr(value_array);

			Compact_Leaf_Info leaf_info;
			for (int n = 0; n < Label_Num / 2; n++) {
				leaf_info.uint_array[n] = tree_array[tree_n].memory_sequence[ptr + n].uint_value;
			}

			for (int i = 0; i < Label_Num; i++) {
				count_array[tree_n * Label_Num + i] = (int)leaf_info.count[i];
			}
		}

		for (int tree_n = 0; tree_n < tree_array.size(); tree_n++) {
			float leaf_sum = 0.0;
			int start_ptr = tree_n * Label_Num;
			for (int i = 0; i < Label_Num; i++) {
				leaf_sum += (float)count_array[start_ptr + i];
			}

			if (leaf_sum > 0) {
				for (int i = 0; i < Label_Num; i++) {
					sum_array[i] += (float)count_array[start_ptr + i] / leaf_sum;
				}
			}
		}
		delete[] count_array;
	}

	int predict(const float* value_array)
	{
		vector<float> sum_array;
		for (int n = 0; n < Label_Num; n++) sum_array.push_back(0.0);

		sum_count(value_array, sum_array);

		auto max_it = max_element(sum_array.begin(), sum_array.end());
		return (int)(distance(sum_array.begin(), max_it));
	}

	int predict(float* value_matrix[], int value_array_num)
	{
		vector<float> sum_array;
		for (int n = 0; n < Label_Num; n++) sum_array.push_back(0.0);

		for (int n = 0; n < value_array_num; n++) {
			sum_count(value_matrix[n], sum_array);
		}

		auto max_it = max_element(sum_array.begin(), sum_array.end());
		return (int)(distance(sum_array.begin(), max_it));
	}

	void project(const int tree_num, const int partition_n, const float* value_array, float* pattern, const int pattern_dimension_num)
	{
		vector<int> ptr_array;
		for (int n = 0; n < tree_num; n++) ptr_array.push_back(0);

#pragma omp parallel for num_threads(8)
		for (int n = 0; n < tree_num; n++) {
			int tree_n = partition_n * tree_num + n;
			ptr_array[n] = tree_array[tree_n].Get_Leaf_Ptr(value_array);
		}

		memset(pattern, 0, pattern_dimension_num * sizeof(float));
		for (int n = 0; n < tree_num; n++) {
			int ptr = ptr_array[n];
			int tree_n = partition_n * tree_num + n;
			int pattern_label_num = tree_array[tree_n].memory_sequence[ptr++].int_value;

			for (int label_n = 0; label_n < pattern_label_num; label_n++) {
				int label = tree_array[tree_n].memory_sequence[ptr++].int_value;
				int SIZE = pattern_dimension_num / Label_Num;
				for (int i = 0; i < SIZE; i++) {
					pattern[label * SIZE + i] += (float)(tree_array[tree_n].memory_sequence[ptr++].float_value);
				}
			}
		}
	}
};

static void Simple_Random_Forest()
{
	Random_Forest recognition_forest;
	recognition_forest.Initialize(Tree_Num_for_Recognition);

	if (Load_Data("data_train.bin", &g_Training_Data, g_Sample_Num, Dimension_Num) != 0) exit(-1);
	if (Load_Label("label_train.bin", &g_Training_Label, g_Sample_Num) != 0) exit(-1);

#pragma region Recognition
	//-----------------------------------------------------------------------------------------------
	{
		if (Do_Training_Recognition) {

			g_Input_Data = g_Training_Data;
			g_Input_Label = g_Training_Label;
			g_Input_Dimension_Num = Dimension_Num;
			g_RF_Type = RF_Type::Recognition;
			g_Query_Ptr_Type = Query_Ptr_Type::Dimension_Ptr;

			auto start = chrono::system_clock::now();
			recognition_forest.fit();
			auto end = chrono::system_clock::now();

			float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
			printf("Training %.1f sec\n", elapsed / 1000.0);
		}
		else {
			recognition_forest.load_from_file("tree");
		}
	}
	//-----------------------------------------------------------------------------------------------
#pragma endregion

#pragma region Evaluation
//-----------------------------------------------------------------------------------------------
	int wrong_count = 0;
	if (Load_Data("data_test.bin", &g_Input_Data, g_Sample_Num, AREA_SIZE) != 0) exit(-1);
	if (Load_Label("label_test.bin", &g_Input_Label, g_Sample_Num) != 0) exit(-1);

	wrong_count = 0;
	for (int n = 0; n < g_Sample_Num; n++) {
		g_RF_Type = RF_Type::Recognition;
		g_Query_Ptr_Type = Query_Ptr_Type::Dimension_Ptr;
		g_Input_Dimension_Num = Dimension_Num;

		int answer = recognition_forest.predict(g_Input_Data + n * g_Input_Dimension_Num);
		if (answer != (int)g_Input_Label[n]) wrong_count++;
	}
	printf("Test wrong_count = %d - %d %lf\n", wrong_count, g_Sample_Num, (float)wrong_count / (float)g_Sample_Num);
	//-----------------------------------------------------------------------------------------------
#pragma endregion
}

static void Single_ES_Forest()
{
	Random_Forest projection_forest_1;
	projection_forest_1.Initialize(Tree_Num_for_Projection_1 * Partition_Num);

	Random_Forest recognition_forest;
	recognition_forest.Initialize(Tree_Num_for_Recognition);

	if (Load_Data("data_train.bin", &g_Training_Data, g_Sample_Num, Dimension_Num) != 0) exit(-1);
	if (Load_Label("label_train.bin", &g_Training_Label, g_Sample_Num) != 0) exit(-1);

	Set_g_Ptr_Arrays();

#pragma region Projection_1
	//-----------------------------------------------------------------------------------------------
	{
		if (Do_Training_Projection_1) {

			g_Input_Data = g_Training_Data;
			g_Input_Label = g_Training_Label;
			g_Input_Dimension_Num = Dimension_Num;
			g_RF_Type = RF_Type::Projection_1;
			g_Query_Ptr_Type = Query_Ptr_Type::Dimension_Ptr;

			auto start = chrono::system_clock::now();
			projection_forest_1.fit();
			auto end = chrono::system_clock::now();

			float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
			printf("Training %.1f sec\n", elapsed / 1000.0);

			FILE* out = nullptr;
			int sample_num = g_Sample_Num;
			int dimension_num = Dimension_Num_1;

			vector<int> output_label_array;
			if (fopen_s(&out, "data_train_projected_1.bin", "wb") == 0) {
				fwrite(&sample_num, sizeof(int), 1, out);
				fwrite(&dimension_num, sizeof(int), 1, out);

				for (int partition_n = 0; partition_n < Partition_Num; partition_n++) {
					for (auto ptr : g_Ptr_Arrays[partition_n]) {
						float* value_array = g_Input_Data + (ptr * g_Input_Dimension_Num);
						int label = g_Input_Label[ptr];
						float float_value;
						float pattern[Dimension_Num_1];
						projection_forest_1.project(Tree_Num_for_Projection_1, partition_n, value_array, pattern, dimension_num);

						for (int i = 0; i < dimension_num; i++) {
							float_value = (float)pattern[i];
							fwrite(&float_value, sizeof(float), 1, out);
						}
						output_label_array.push_back(label);
					}
				}
				fclose(out);
			}

			if (fopen_s(&out, "label_train_projected_1.bin", "wb") == 0) {
				fwrite(&sample_num, sizeof(int), 1, out);
				for (auto label : output_label_array) {
					unsigned char _out = label;
					fwrite(&_out, 1, 1, out);
				}
				fclose(out);
			}
		}
		else {
			projection_forest_1.load_from_file("tree_projection_1");
		}
	}
	printf("----- Projection_1 Finish -----\n");
	//-----------------------------------------------------------------------------------------------
#pragma endregion

#pragma region Recognition
	//-----------------------------------------------------------------------------------------------
	{
		if (Do_Training_Recognition) {

			if (Load_Data("data_train_projected_1.bin", &g_Training_Data_1, g_Sample_Num, Dimension_Num_1) != 0) exit(-1);
			if (Load_Label("label_train_projected_1.bin", &g_Training_Label_1, g_Sample_Num) != 0) exit(-1);

			g_Input_Data = g_Training_Data_1;
			g_Input_Label = g_Training_Label_1;
			g_Input_Dimension_Num = Dimension_Num_1;
			g_RF_Type = RF_Type::Recognition;
			g_Query_Ptr_Type = Query_Ptr_Type::Instance_Ptr;

			auto start = chrono::system_clock::now();
			recognition_forest.fit();
			auto end = chrono::system_clock::now();

			float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
			printf("Training %.1f sec\n", elapsed / 1000.0);
		}
		else {
			recognition_forest.load_from_file("tree");
		}
	}
	//-----------------------------------------------------------------------------------------------
#pragma endregion

#pragma region Evaluation
//-----------------------------------------------------------------------------------------------
	int wrong_count = 0;
	if (Do_Training_Recognition == false) {
		if (Load_Data("data_train_projected_1.bin", &g_Training_Data_1, g_Sample_Num, Dimension_Num_1) != 0) exit(-1);
		if (Load_Label("label_train_projected_1.bin", &g_Training_Label_1, g_Sample_Num) != 0) exit(-1);
	}

	if (Load_Data("data_test.bin", &g_Training_Data, g_Sample_Num, AREA_SIZE) != 0) exit(-1);
	if (Load_Label("label_test.bin", &g_Training_Label, g_Sample_Num) != 0) exit(-1);

	wrong_count = 0;
	float* value_matrix[Partition_Num];
	for (int n = 0; n < Partition_Num; n++) value_matrix[n] = new float[Dimension_Num_1];

	for (int n = 0; n < g_Sample_Num; n++) {
		for (int partition_n_1 = 0; partition_n_1 < Partition_Num; partition_n_1++) {
			g_RF_Type = RF_Type::Projection_1;
			g_Query_Ptr_Type = Query_Ptr_Type::Dimension_Ptr;
			projection_forest_1.project(Tree_Num_for_Projection_1, partition_n_1, g_Training_Data + (n * Dimension_Num), value_matrix[partition_n_1], Dimension_Num_1);
		}

		g_RF_Type = RF_Type::Recognition;
		g_Query_Ptr_Type = Query_Ptr_Type::Instance_Ptr;
		g_Input_Data = g_Training_Data_1;
		g_Input_Dimension_Num = Dimension_Num_1;

		int answer = recognition_forest.predict(value_matrix, Partition_Num);
		if (answer != (int)g_Training_Label[n]) wrong_count++;
	}
	printf("Test wrong_count = %d - %d %lf\n", wrong_count, g_Sample_Num, (float)wrong_count / (float)g_Sample_Num);

	for (int n = 0; n < Partition_Num; n++) delete[] value_matrix[n];
	//-----------------------------------------------------------------------------------------------
#pragma endregion
}

static void Double_ES_Forest()
{
	Random_Forest projection_forest_1;
	projection_forest_1.Initialize(Tree_Num_for_Projection_1 * Partition_Num);

	Random_Forest projection_forest_2;
	projection_forest_2.Initialize(Tree_Num_for_Projection_2 * Partition_Num);

	Random_Forest recognition_forest;
	recognition_forest.Initialize(Tree_Num_for_Recognition);


	if (Load_Data("data_train.bin", &g_Training_Data, g_Sample_Num, Dimension_Num) != 0) exit(-1);
	if (Load_Label("label_train.bin", &g_Training_Label, g_Sample_Num) != 0) exit(-1);

	Set_g_Ptr_Arrays();

#pragma region Projection_1
	//-----------------------------------------------------------------------------------------------
	{
		if (Do_Training_Projection_1) {
			g_Input_Data = g_Training_Data;
			g_Input_Label = g_Training_Label;
			g_Input_Dimension_Num = Dimension_Num;
			g_RF_Type = RF_Type::Projection_1;
			g_Query_Ptr_Type = Query_Ptr_Type::Dimension_Ptr;

			auto start = chrono::system_clock::now();
			projection_forest_1.fit();
			auto end = chrono::system_clock::now();

			float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
			printf("Training %.1f sec\n", elapsed / 1000.0);

			FILE* out = nullptr;
			int sample_num = g_Sample_Num;
			int dimension_num = Dimension_Num_1;
			vector<int> output_label_array;
			if (fopen_s(&out, "data_train_projected_1.bin", "wb") == 0) {
				fwrite(&sample_num, sizeof(int), 1, out);
				fwrite(&dimension_num, sizeof(int), 1, out);
				for (int partition_n = 0; partition_n < Partition_Num; partition_n++) {
					for (auto ptr : g_Ptr_Arrays[partition_n]) {
						float* value_array = g_Input_Data + (ptr * g_Input_Dimension_Num);
						int label = g_Input_Label[ptr];
						float float_value;
						float pattern[Dimension_Num_1];
						projection_forest_1.project(Tree_Num_for_Projection_1, partition_n, value_array, pattern, dimension_num);

						for (int i = 0; i < dimension_num; i++) {
							float_value = (float)pattern[i];
							fwrite(&float_value, sizeof(float), 1, out);
						}
						output_label_array.push_back(label);
					}
				}
				fclose(out);
			}

			if (fopen_s(&out, "label_train_projected_1.bin", "wb") == 0) {
				fwrite(&sample_num, sizeof(int), 1, out);
				for (auto label : output_label_array) {
					unsigned char _out = label;
					fwrite(&_out, 1, 1, out);
				}
				fclose(out);
			}
		}
		else {
			projection_forest_1.load_from_file("tree_projection_1");
		}
	}
	printf("----- Projection_1 Finish -----\n");
	//-----------------------------------------------------------------------------------------------
#pragma endregion

#pragma region Projection_2
//-----------------------------------------------------------------------------------------------
	{
		if (Do_Training_Projection_2) {

			if (Load_Data("data_train_projected_1.bin", &g_Training_Data_1, g_Sample_Num, Dimension_Num_1) != 0) exit(-1);
			if (Load_Label("label_train_projected_1.bin", &g_Training_Label_1, g_Sample_Num) != 0) exit(-1);

			g_Input_Data = g_Training_Data_1;
			g_Input_Label = g_Training_Label_1;
			g_Input_Dimension_Num = Dimension_Num_1;
			g_RF_Type = RF_Type::Projection_2;
			g_Query_Ptr_Type = Query_Ptr_Type::Instance_Ptr;

			auto start = chrono::system_clock::now();
			projection_forest_2.fit();
			auto end = chrono::system_clock::now();

			float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
			printf("Training %.1f sec\n", elapsed / 1000.0);

			FILE* out = nullptr;
			int sample_num = g_Sample_Num;
			int dimension_num = Dimension_Num_2;
			vector<int> output_label_array;
			if (fopen_s(&out, "data_train_projected_2.bin", "wb") == 0) {
				fwrite(&sample_num, sizeof(int), 1, out);
				fwrite(&dimension_num, sizeof(int), 1, out);
				for (int partition_n = 0; partition_n < Partition_Num; partition_n++) {
					for (auto ptr : g_Ptr_Arrays[partition_n]) {
						float* value_array = g_Input_Data + (ptr * g_Input_Dimension_Num);
						int label = g_Input_Label[ptr];
						float float_value;
						float pattern[Dimension_Num_2];
						projection_forest_2.project(Tree_Num_for_Projection_2, partition_n, value_array, pattern, dimension_num);

						for (int i = 0; i < dimension_num; i++) {
							float_value = (float)pattern[i];
							fwrite(&float_value, sizeof(float), 1, out);
						}
						output_label_array.push_back(label);
					}
				}
				fclose(out);
			}
			if (fopen_s(&out, "label_train_projected_2.bin", "wb") == 0) {
				fwrite(&sample_num, sizeof(int), 1, out);
				for (auto label : output_label_array) {
					unsigned char _out = label;
					fwrite(&_out, 1, 1, out);
				}
				fclose(out);
			}
		}
		else {
			projection_forest_2.load_from_file("tree_projection_2");
		}
	}
	printf("----- Projection_2 Finish -----\n");
	//-----------------------------------------------------------------------------------------------
#pragma endregion

#pragma region Recognition
//-----------------------------------------------------------------------------------------------
	{
		if (Do_Training_Recognition) {

			if (Load_Data("data_train_projected_2.bin", &g_Training_Data_2, g_Sample_Num, Dimension_Num_2) != 0) exit(-1);
			if (Load_Label("label_train_projected_2.bin", &g_Training_Label_2, g_Sample_Num) != 0) exit(-1);

			g_Input_Data = g_Training_Data_2;
			g_Input_Label = g_Training_Label_2;
			g_Input_Dimension_Num = Dimension_Num_2;
			g_RF_Type = RF_Type::Recognition;
			g_Query_Ptr_Type = Query_Ptr_Type::Instance_Ptr;

			auto start = chrono::system_clock::now();
			recognition_forest.fit();
			auto end = chrono::system_clock::now();

			float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
			printf("Training %.1f sec\n", elapsed / 1000.0);
		}
		else {
			recognition_forest.load_from_file("tree");
		}
	}
	//-----------------------------------------------------------------------------------------------
#pragma endregion

#pragma region Evaluation
//-----------------------------------------------------------------------------------------------
	int wrong_count = 0;
	if (Do_Training_Projection_2 == false) {
		if (Load_Data("data_train_projected_1.bin", &g_Training_Data_1, g_Sample_Num, Dimension_Num_1) != 0) exit(-1);
		if (Load_Label("label_train_projected_1.bin", &g_Training_Label_1, g_Sample_Num) != 0) exit(-1);
	}

	if (Do_Training_Recognition == false) {
		if (Load_Data("data_train_projected_2.bin", &g_Training_Data_2, g_Sample_Num, Dimension_Num_2) != 0) exit(-1);
		if (Load_Label("label_train_projected_2.bin", &g_Training_Label_2, g_Sample_Num) != 0) exit(-1);
	}

	if (Load_Data("data_test.bin", &g_Training_Data, g_Sample_Num, AREA_SIZE) != 0) exit(-1);
	if (Load_Label("label_test.bin", &g_Training_Label, g_Sample_Num) != 0) exit(-1);

	wrong_count = 0;
	float* value_matrix_1[Partition_Num];
	for (int n = 0; n < Partition_Num; n++) value_matrix_1[n] = new float[Dimension_Num_1];
	float* value_matrix_2[Partition_Num * Partition_Num];
	for (int n = 0; n < Partition_Num * Partition_Num; n++) value_matrix_2[n] = new float[Dimension_Num_2];

	for (int n = 0; n < g_Sample_Num; n++) {
		for (int partition_n_1 = 0; partition_n_1 < Partition_Num; partition_n_1++) {
			g_RF_Type = RF_Type::Projection_1;
			g_Query_Ptr_Type = Query_Ptr_Type::Dimension_Ptr;
			projection_forest_1.project(Tree_Num_for_Projection_1, partition_n_1, g_Training_Data + (n * Dimension_Num), value_matrix_1[partition_n_1], Dimension_Num_1);

			for (int partition_n_2 = 0; partition_n_2 < Partition_Num; partition_n_2++) {
				int partition_n = Partition_Num * partition_n_1 + partition_n_2;
				g_RF_Type = RF_Type::Projection_2;
				g_Query_Ptr_Type = Query_Ptr_Type::Instance_Ptr;
				g_Input_Data = g_Training_Data_1;
				g_Input_Dimension_Num = Dimension_Num_1;
				projection_forest_2.project(Tree_Num_for_Projection_2, partition_n_2, value_matrix_1[partition_n_1], value_matrix_2[partition_n], Dimension_Num_2);
			}
		}

		g_RF_Type = RF_Type::Recognition;
		g_Query_Ptr_Type = Query_Ptr_Type::Instance_Ptr;
		g_Input_Data = g_Training_Data_2;
		g_Input_Dimension_Num = Dimension_Num_2;

		int answer = recognition_forest.predict(value_matrix_2, Partition_Num * Partition_Num);
		if (answer != (int)g_Training_Label[n]) wrong_count++;
	}
	printf("Test wrong_count = %d - %d %lf\n", wrong_count, g_Sample_Num, (float)wrong_count / (float)g_Sample_Num);

	for (int n = 0; n < Partition_Num; n++) delete[] value_matrix_1[n];
	for (int n = 0; n < Partition_Num; n++) delete[] value_matrix_2[n];
	//-----------------------------------------------------------------------------------------------
#pragma endregion
}

void Select_and_Out()
{
	Load_Data("data_train.bin", &g_Training_Data, g_Sample_Num, Dimension_Num);
	Load_Label("label_train.bin", &g_Training_Label, g_Sample_Num);

	mt19937_64 random_engine(0);
	vector<int> ptr_array;
	for (int n = 0; n < g_Sample_Num; n++) ptr_array.push_back(n);
	shuffle(ptr_array.begin(), ptr_array.end(), random_engine);

	int sample_num = g_Sample_Num / 10;
	vector<unsigned char> label_array;
	FILE* out = nullptr;
	float float_value;
	if (fopen_s(&out, "data_train_new.bin", "wb") == 0) {
		fwrite(&sample_num, sizeof(int), 1, out);
		fwrite(&Dimension_Num, sizeof(int), 1, out);
		for (int n = 0; n < sample_num; n++) {
			int ptr = ptr_array[n];
			for (int i = 0; i < Dimension_Num; i++) {
				float_value = g_Training_Data[ptr * Dimension_Num + i];
				fwrite(&float_value, sizeof(float), 1, out);
			}
			label_array.push_back(g_Training_Label[ptr]);
		}
		fclose(out);
	}
	if (fopen_s(&out, "label_train_new.bin", "wb") == 0) {
		fwrite(&sample_num, sizeof(int), 1, out);
		for (auto label : label_array) {
			fwrite(&label, 1, 1, out);
		}
		fclose(out);
	}
}

int main()
{
	switch (g_Program_Type)
	{
	case (Program_Type::Simple_Random_Forest):
		Simple_Random_Forest();
		break;
	case (Program_Type::Single_ES_Forest):
		Single_ES_Forest();
		break;
	case (Program_Type::Double_ES_Forest):
		Double_ES_Forest();
		break;
	case (Program_Type::Just_Select_Samples):
		Select_and_Out();
		break;
	}

	return 0;
}
