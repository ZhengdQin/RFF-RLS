%输入U为列向量，W是权重列向量，lambda是遗忘因子,e是当前误差，P是逆相关矩阵
function [W_new,P_new]=LSE(u,W,e,P,lambda)

S = P*u;
K = S./(lambda+u'*S);
W_new = W +K.*e;
P_new = P./lambda-K*u'*P./lambda;

end