function [seg,loglik]=bmass(in,out,odds,varargin) 
%BMASS    Bayesian model-based agglomerative sequence segmentation
%   
%   BMASS partitions a sequence of real-valued input-output data into
%   non-overlapping segments. The segment boundaries are chosen under the
%   assumption that, within each segment, the data follow a multi-variate
%   linear model.
%   
%   Sequence segmentation proceeds by greedily merging pairs of consecutive
%   segments. Initially, all the data are placed in separate segments. In
%   each iteration, a single pair of segments is merged. The decision on
%   which pair of segments to merge is based on the log-likelihood ratio of
%   the merge hypothesis. The merging process continues until either the
%   log-likelihood ratio falls below zero, or all segments have been
%   merged into one. At this point the process terminates, returning the
%   final segmentation.
%   
%   Usage: [Seg,LogLik] = BMASS(X,Y,Odds)
%   
%   Inputs:  X      - Sequence of input data  £¡£¡£¡
%            Y      - Sequence of output data  £¡£¡£¡
%            Odds   - Merging odds  £¡£¡£¡(hazard function?)
%   
%   Outputs: Seq    - Segment indices £¡£¡£¡
%            LogLik - Log-likelihood ratios  £¡£¡£¡
%   
%   Inputs X and Y must be matrices with the same number of rows. Each pair
%   of rows X(k,:) and Y(k,:) is a corresponding pair of input-output data.
%   The algorithm assumes that, within each segment, the data follow a
%   linear-Gaussian law. In other words, that the conditional distribution
%   of Y(k,:) given X(k,:) for all k in a given segment is Gaussian with
%   conditional mean X(k,:)*A and covariance Q for some A, Q.
%   
%   Input Odds must be a scalar greater than one. To some extent, its value
%   affects the outcome of the algorithm, by biasing the result towards
%   finer or coarser segmentations. In practice, Odds should be roughly
%   equal to the number of data divided by the total number of segments.
%   
%   Output Seq is a cell array with one cell for each segment found. Cell
%   Seq{i} contains the range of indices of the corresponding segment. That
%   is, Seq{i} = Bound(i)+1:Bound(i+1), where Bound is a vector containing
%   the segment boundaries identified by the algorithm.
%   
%   Output LogLik is a vector with as many elements as iterations. Each
%   element LogLik(i) contains the log-likelihood ratio of the optimal
%   merge hypothesis at that iteration. The last element, LogLik(end), is
%   always negative, except if no segment boundaries are found.
%   
%   In addition to the inputs, the syntax 
%   
%   [...] = BMASS(...,'HyperParam',Val,...)
%   
%   accepts a list of hyper-parameters in the form of property/value pairs.
%   These are the Gain, Scale, Noise and Shape hyper-parameters of the
%   prior Gaussian-Wishart distribution over the segment-specific
%   regression parameters (A and Q).
%   
%   Copyright (c) 2014 Gabriel Agamennoni.

% Check number of arguments.
% so input should >=2 and out should <=2
% nargin
% size(in)
% size(out)

if nargin()<2
    error('BMASS:NotEnoughInputs',...
        'Not enough inputs.')
end
if nargout()>2
    error('BMASS:TooManyOutputs',...
        'Too many outputs.')
end

% Check arguments and scan hyper-parameters from property-value pairs.
[gain,scale,noise,shape]=check(in,out,odds,varargin{:}); 

% Store number of inputs, outputs and observations.
[numobs,numin]=size(in);
[~,numout]=size(out);

% Allocate space for segments and log-likelihood ratios.
seg=cell(numobs,1);
loglik=zeros(numobs-1,1);

% Allocate space for sufficient statistics.
stat=zeros(numin+numout,numin+numout,numobs);
weight=zeros(numobs,1);

% Allocate space for log-normalization constants.
nomerge=zeros(numobs,1);
merge=zeros(numobs-1,1);

% Evaluate prior log-normalization constant.
const=eval(gain,scale,noise,shape,zeros(numin+numout,numin+numout),0);

% Initialize by placing data in separate segments.
aux={[],[]};
for i=1:numobs
    
    % Initialize segment.
    seg{i}=i;
    
    % Initialize sufficient statistics.
    [stat(:,:,i),weight(i)]=init(in(i,:),out(i,:));
    
    % Evaluate log-normalization constants for merge/non-merge hypotheses.
    nomerge(i)=eval(gain,scale,noise,shape,stat(:,:,i),weight(i));
    if i>1
        [aux{:}]=comb(stat(:,:,i-1:i),weight(i-1:i));
        merge(i-1)=eval(gain,scale,noise,shape,aux{:});
    end
    
end

% Iteratively merge pairs of consecutive segments.
odds=log(odds);
for i=1:numobs-1
    
    % Find pair of consecutive segments with highest log-likelihood ratio.
    [loglik(i),j]=max(odds+const+merge-nomerge(1:end-1)-nomerge(2:end));
    
    % Check if no further improvement possible.
    if loglik(i)<0
        break
    end
    
    % Merge segments.
    seg{j}=[seg{j},seg{j+1}];
    
    % Merge sufficient statistics and evaluate log-normalization constant.
    [stat(:,:,j),weight(j)]=comb(stat(:,:,j:j+1),weight(j:j+1));
    nomerge(j)=eval(gain,scale,noise,shape,stat(:,:,j),weight(j));
    
    % Delete segment.
    seg(j+1)=[];
    
    % Delete sufficient statistics.
    stat(:,:,j+1)=[];
    weight(j+1)=[];
    
    % Delete log-normalization constants.
    nomerge(j+1)=[];
    merge(j)=[];
    
    % Update log-normalization constants for next iteration.
    if j>1
        [aux{:}]=comb(stat(:,:,j-1:j),weight(j-1:j));
        merge(j-1)=eval(gain,scale,noise,shape,aux{:});
    end
    if j<numobs-i
        [aux{:}]=comb(stat(:,:,j:j+1),weight(j:j+1));
        merge(j)=eval(gain,scale,noise,shape,aux{:});
    end
    
end
loglik=loglik(1:i);

end



function [stat,weight]=init(in,out)

% Initialize sufficient statistics.
stat=[in,out]'*[in,out];
weight=1;

end



function [stat,weight]=comb(stat,weight)

% Combine sufficient statistics.
stat=sum(stat,3);
weight=sum(weight);

end



function const=eval(gain,scale,noise,shape,stat,weight)

% Store number of inputs and outputs.
[numin,numout]=size(gain);

% Store indices to upper/lower matrix blocks.
i=1:numin;
j=numin+1:numin+numout;

% Increment weight.
weight=weight+shape;

% Build and factorize outer-product matrix.
fact=scale*gain;
fact=chol([scale+stat(i,i),fact+stat(i,j);fact'+stat(j,i),...
    shape*noise+gain'*fact+stat(j,j)]/weight,'lower');

% Evaluate log-normalization constant of posterior distribution.
const=-numout*sum(log(diag(chol(scale+stat(i,i)))))-...
    weight*sum(log(diag(fact(j,j))))-numout*(weight/2)*log(weight/2)+...
    sum(gammaln((weight+1-(1:numout))/2));

end



function [gain,scale,noise,shape]=check(in,out,odds,varargin)

% Initialize argument counter.
arg=0;

% Check input data.
arg=arg+1;
if ~isnumeric(in)
    error('BMASS:BadInputClass', ...
        'Input %d must be numeric.',arg)
end
if ~isreal(in)
    error('BMASS:BadInputClass', ...
        'Input %d must be real.',arg)
end
if isempty(in)
    error('BMASS:BadInputSize', ...
        'Input %d must be non-empty.',arg)
end
if ndims(in)>2
    error('BMASS:BadInputSize', ...
        'Input %d must be a matrix.',arg)
end
if any(isnan(in(:))|isinf(in(:)))
    error('BMASS:BadInputValue', ...
        'Input %d must contain finite numbers.',arg)
end

% Store number of inputs and number of observations.
[numobs,numin]=size(in);

% Check output data.
arg=arg+1;
if ~isnumeric(out)
    error('BMASS:BadInputClass', ...
        'Input %d must be numeric.',arg)
end
if ~isreal(out)
    error('BMASS:BadInputClass', ...
        'Input %d must be real.',arg)
end
if isempty(out)
    error('BMASS:BadInputSize', ...
        'Input %d must be non-empty.',arg)
end
if ndims(out)>2
    error('BMASS:BadInputSize', ...
        'Input %d must be a matrix.',arg)
end
if size(out,1)~=numobs
    error('BMASS:BadInputSize', ...
        'Input %d must have %d row(s).',arg,numobs)
end
if any(isnan(out(:))|isinf(out(:)))
    error('BMASS:BadInputValue', ...
        'Input %d must contain finite numbers.',arg)
end

% Store number of outputs.
[~,numout]=size(out);

% Check merging odds.
arg=arg+1;
if ~isnumeric(odds)
    error('BMASS:BadInputClass',...
        'Input %d must be numeric.',arg)
end
if ~isreal(odds)
    error('BMASS:BadInputClass',...
        'Input %d must be real.',arg)
end
if isempty(odds)
    error('BMASS:BadInputSize',...
        'Input %d must be non-empty.',arg)
end
if ndims(odds)>2||numel(odds)>1
    error('BMASS:BadInputSize',...
        'Input %d must be a scalar.',arg)
end
if isnan(odds)||isinf(odds)
    error('BMASS:BadInputValue',...
        'Input %d must contain a finite number.',arg)
end
if odds<=1
    error('BMASS:BadInputValue',...
        'Input %d must contain a value greater than %d.',arg,1)
end

% Store default hyper-parameters.
gain=zeros(numin,numout);
scale=eye(numin);
noise=eye(numout,numout);
shape=numout;

% Scan hyper-parameters from property-value pairs.
for i=1:2:numel(varargin)
    
    % Check for early return.
    if numel(varargin)<i+1
        warning('BMASS:IgnoringLastInput', ...
            'Ignoring last input.')
        break
    end
    
    % Check property.
    arg=arg+1;
    if ~ischar(varargin{i})
        error('BMASS:BadInputClass', ...
            'Input %d must be a string.',arg)
    end
    if isempty(varargin{i})
        error('BMASS:BadInputSize', ...
            'Input %d must be non-empty.',arg)
    end
    if ndims(varargin{i})>2||size(varargin{i},1)>1
        error('BMASS:BadInputSize', ...
            'Input %d must be a string.',arg)
    end
    
    % Match property.
    switch lower(varargin{i})
        case 'gain'
            
            % Check gain hyper-parameters.
            arg=arg+1;
            if ~isnumeric(varargin{i+1})
                error('BMASS:BadInputClass',...
                    'Input %d must be numeric.',arg)
            end
            if ~isreal(varargin{i+1})
                error('BMASS:BadInputClass',...
                    'Input %d must be real.',arg)
            end
            if isempty(varargin{i+1})
                error('BMASS:BadInputSize',...
                    'Input %d must be non-empty.',arg)
            end
            if ndims(varargin{i+1})>2
                error('BMASS:BadInputSize',...
                    'Input %d must be a matrix.',arg)
            end
            if size(varargin{i+1},1)~=numin
                error('BMASS:BadInputSize',...
                    'Input %d must have %d row(s).',arg,numin)
            end
            if size(varargin{i+1},2)~=numout
                error('BMASS:BadInputSize',...
                    'Input %d must have %d column(s).',arg,numout)
            end
            if any(isnan(varargin{i+1}(:))|isinf(varargin{i+1}(:)))
                error('BMASS:BadInputValue',...
                    'Input %d must contain finite numbers.',arg)
            end
            
            % Set gain hyper-parameters.
            gain=varargin{i+1};
            
        case 'scale'
            
            % Check scale hyper-parameters.
            arg=arg+1;
            if ~isnumeric(varargin{i+1})
                error('BMASS:BadInputClass',...
                    'Input %d must be numeric.',arg)
            end
            if ~isreal(varargin{i+1})
                error('BMASS:BadInputClass',...
                    'Input %d must be real.',arg)
            end
            if isempty(varargin{i+1})
                error('BMASS:BadInputSize',...
                    'Input %d must be non-empty.',arg)
            end
            if ndims(varargin{i+1})>2
                error('BMASS:BadInputSize',...
                    'Input %d must be a matrix.',arg)
            end
            if size(varargin{i+1},1)~=numin
                error('BMASS:BadInputSize',...
                    'Input %d must have %d row(s).',arg,numin)
            end
            if size(varargin{i+1},2)~=numin
                error('BMASS:BadInputSize',...
                    'Input %d must have %d column(s).',arg,numin)
            end
            if any(isnan(varargin{i+1}(:))|isinf(varargin{i+1}(:)))
                error('BMASS:BadInputValue',...
                    'Input %d must contain finite numbers.',arg)
            end
            asym=varargin{i+1}-varargin{i+1}';
            if any(abs(asym(:))>eps()*numin)
                error('BMASS:BadInputValue',...
                    'Input %d must contain a symmetric matrix.',arg)
            end
            [~,sing]=chol(varargin{i+1});
            if sing>0
                error('BMASS:BadInputValue',...
                    'Input %d must contain a positive-definite matrix.',arg)
            end
            
            % Set scale hyper-parameters.
            scale=varargin{i+1};
            
        case 'noise'
            
            % Check noise hyper-parameters.
            arg=arg+1;
            if ~isnumeric(varargin{i+1})
                error('BMASS:BadInputClass',...
                    'Input %d must be numeric.',arg)
            end
            if ~isreal(varargin{i+1})
                error('BMASS:BadInputClass',...
                    'Input %d must be real.',arg)
            end
            if isempty(varargin{i+1})
                error('BMASS:BadInputSize',...
                    'Input %d must be non-empty.',arg)
            end
            if ndims(varargin{i+1})>2
                error('BMASS:BadInputSize',...
                    'Input %d must be a matrix.',arg)
            end
            if size(varargin{i+1},1)~=numout
                error('BMASS:BadInputSize',...
                    'Input %d must have %d row(s).',arg,numout)
            end
            if size(varargin{i+1},2)~=numout
                error('BMASS:BadInputSize',...
                    'Input %d must have %d column(s).',arg,numout)
            end
            if any(isnan(varargin{i+1}(:))|isinf(varargin{i+1}(:)))
                error('BMASS:BadInputValue',...
                    'Input %d must contain finite numbers.',arg)
            end
            asym=varargin{i+1}-varargin{i+1}';
            if any(abs(asym(:))>eps()*numout)
                error('BMASS:BadInputValue',...
                    'Input %d must contain a symmetric matrix.',arg)
            end
            [~,sing]=chol(varargin{i+1});
            if sing>0
                error('BMASS:BadInputValue',...
                    'Input %d must contain a positive-definite matrix.',arg)
            end
            
            % Set noise hyper-parameters.
            noise=varargin{i+1};
            
        case 'shape'
            
            % Check shape hyper-parameter.
            arg=arg+1;
            if ~isnumeric(varargin{i+1})
                error('BMASS:BadInputClass',...
                    'Input %d must be numeric.',arg)
            end
            if ~isreal(varargin{i+1})
                error('BMASS:BadInputClass',...
                    'Input %d must be real.',arg)
            end
            if isempty(varargin{i+1})
                error('BMASS:BadInputSize',...
                    'Input %d must be non-empty.',arg)
            end
            if ndims(varargin{i+1})>2||numel(varargin{i+1})>1
                error('BMASS:BadInputSize',...
                    'Input %d must be a scalar.',arg)
            end
            if isnan(varargin{i+1})||isinf(varargin{i+1})
                error('BMASS:BadInputValue',...
                    'Input %d must contain a finite number.',arg)
            end
            if varargin{i+1}<=numout-1
                error('BMASS:BadInputValue',...
                    'Input %d must contain a value greater than %d.',...
                        arg,numout-1)
            end
            
            % Set shape hyper-parameter.
            shape=varargin{i+1};
            
        otherwise
            error('BMASS:BadProperty', ...
                'Input %d is not a valid option.',arg)
    end
    
end

end