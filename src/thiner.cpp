#include <cmath>

int funcNc8(int *b)
//�˵����ͨ�Լ��
{
	int n_odd[4] = { 1, 3, 5, 7 };  //������
	int i, j, sum, d[10];

	for (i = 0; i <= 9; i++) {
		j = i;
		if (i == 9) j = 1;
		if (abs(*(b + j)) == 1)
		{
			d[i] = 1;
		}
		else
		{
			d[i] = 0;
		}
	}
	sum = 0;
	for (i = 0; i < 4; i++)
	{
		j = n_odd[i];
		sum = sum + d[j] - d[j] * d[j + 1] * d[j + 2];
	}
	return (sum);
}

//ϸ���㷨
void hilditchThin(bool* inside_array, bool* output_array, int w, int h, int d)
{
	int GRAY = 128, WHITE = 255, BLACK = 0;
	for (int z = 0; z < d; ++z)
	{
		int *img = new int[w * h];
		for (int i = 0; i < h; i++)
		{

			for (int j = 0; j < w; j++)
			{
				img[j + i * w] = (int)inside_array[j + i * w + z * w * h] * 255;
			}
		}
		int offset[9][2] = { {0,0},{1,0},{1,-1},{0,-1},{-1,-1},
	{-1,0},{-1,1},{0,1},{1,1} };
		//�������ƫ����
		int n_odd[4] = { 1, 3, 5, 7 };
		int px, py;
		int b[9];                      //3*3���ӵĻҶ���Ϣ
		int condition[6];              //1-6�������Ƿ�����
		int counter;                   //��ȥ���ص�����
		int i, x, y, copy, sum;
		do
		{

			counter = 0;

			for (y = 0; y < h; y++)
			{

				for (x = 0; x < w; x++)
				{

					//ǰ����Ϊɾ�������أ�����������Ӧ����ֵΪ-1
					for (i = 0; i < 9; i++)
					{
						b[i] = 0;
						px = x + offset[i][0];
						py = y + offset[i][1];
						if (px >= 0 && px < w &&    py >= 0 && py < h)
						{
							// printf("%d\n", img[py*step+px]);
							if (img[py*w + px] == WHITE)
							{
								b[i] = 1;
							}
							else if (img[py*w + px] == GRAY)
							{
								b[i] = -1;
							}
						}
					}
					for (i = 0; i < 6; i++)
					{
						condition[i] = 0;
					}

					//����1����ǰ����
					if (b[0] == 1) condition[0] = 1;

					//����2���Ǳ߽��
					sum = 0;
					for (i = 0; i < 4; i++)
					{
						sum = sum + 1 - abs(b[n_odd[i]]);
					}
					if (sum >= 1) condition[1] = 1;

					//����3�� �˵㲻��ɾ��
					sum = 0;
					for (i = 1; i <= 8; i++)
					{
						sum = sum + abs(b[i]);
					}
					if (sum >= 2) condition[2] = 1;

					//����4�� �����㲻��ɾ��
					sum = 0;
					for (i = 1; i <= 8; i++)
					{
						if (b[i] == 1) sum++;
					}
					if (sum >= 1) condition[3] = 1;

					//����5�� ��ͨ�Լ��
					if (funcNc8(b) == 1) condition[4] = 1;

					//����6�����Ϊ2�ĹǼ�ֻ��ɾ��1��
					sum = 0;
					for (i = 1; i <= 8; i++)
					{
						if (b[i] != -1)
						{
							sum++;
						}
						else
						{
							copy = b[i];
							b[i] = 0;
							if (funcNc8(b) == 1) sum++;
							b[i] = copy;
						}
					}
					if (sum == 8) condition[5] = 1;

					if (condition[0] && condition[1] && condition[2] && condition[3] && condition[4] && condition[5])
					{
						img[y*w + x] = GRAY; //����ɾ������λGRAY��GRAY��ɾ����ǣ�������Ϣ�Ժ������ص��ж�����
						counter++;
					}
				}
			}

			if (counter != 0)
			{
				for (y = 0; y < h; y++)
				{
					for (x = 0; x < w; x++)
					{
						if (img[y*w + x] == GRAY)
							img[y*w + x] = BLACK;

					}
				}
			}

		} while (counter != 0);
		for (int i = 0; i < h; i++)
		{

			for (int j = 0; j < w; j++)
			{
				if (img[j + i * w] == WHITE)
					output_array[j + i * w + z * w * h] = true;
				else
					output_array[j + i * w + z * w * h] = false;
			}
		}
	}
}