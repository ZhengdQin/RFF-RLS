%����UΪ��������W��Ȩ����������lambda����������,e�ǵ�ǰ��P������ؾ���
function [W_new,P_new]=LSE(u,W,e,P,lambda)

S = P*u;
K = S./(lambda+u'*S);
W_new = W +K.*e;
P_new = P./lambda-K*u'*P./lambda;

end