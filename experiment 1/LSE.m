%����UΪ��������W��Ȩ����������lambda����������,e�ǵ�ǰ��P������ؾ���
%output:U, W: weight vector, lambda: forgetten factor, e:current error,
%P:Inverse correlation matrix
function [W_new,P_new]=LSE(u,W,e,P,lambda)

S = P*u;
K = S./(lambda+u'*S);
W_new = W +K.*e;
P_new = P./lambda-K*u'*P./lambda;

end