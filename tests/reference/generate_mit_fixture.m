% Generate a minimal numerical oracle using the external MIT toolbox.
%
% The toolbox itself is deliberately not vendored. Set these variables before
% invoking MATLAB:
%   LATENTLINKS_MIT_TOOLBOX   directory containing the toolbox .m files
%   LATENTLINKS_FIXTURE_OUT   destination JSON file

toolbox_dir = getenv('LATENTLINKS_MIT_TOOLBOX');
output_file = getenv('LATENTLINKS_FIXTURE_OUT');
if isempty(toolbox_dir) || isempty(output_file)
    error('LATENTLINKS_MIT_TOOLBOX and LATENTLINKS_FIXTURE_OUT are required');
end
addpath(toolbox_dir);

bnet = mk_influence([2 2], [2 2], 'dd');
bnet.Pi{1} = [0.65 0.35];
bnet.Pi{2} = [0.40 0.60];
bnet.A{1,1} = [0.85 0.15; 0.25 0.75];
bnet.A{1,2} = [0.70 0.30; 0.10 0.90];
bnet.A{2,1} = [0.20 0.80; 0.75 0.25];
bnet.A{2,2} = [0.90 0.10; 0.30 0.70];
bnet.B{1} = [0.88 0.12; 0.18 0.82];
bnet.B{2} = [0.80 0.20; 0.25 0.75];
% Toolbox convention is T(source, target), with columns summing to one.
bnet.T = [0.80 0.25; 0.20 0.75];

observations_one_based = [1 1 2 2 1; 2 2 1 2 1];
evidence = num2cell(observations_one_based);
engine = influence_inf_engine(bnet);
[engine, log_likelihood] = enter_evidence(engine, evidence);

fixture.schema_version = 1;
fixture.observations_one_based = observations_one_based;
fixture.initial = cell2mat(bnet.Pi');
fixture.transition_source_target = cat(3, ...
    bnet.A{1,1}, bnet.A{1,2}, bnet.A{2,1}, bnet.A{2,2});
fixture.emission = cat(3, bnet.B{1}, bnet.B{2});
fixture.influence_source_target = bnet.T;
fixture.log_likelihood = log_likelihood;
fixture.gamma_time_by_flat_state = engine.gamma;
fixture.xi_flat_source_by_target = engine.xi;

file_id = fopen(output_file, 'w');
if file_id < 0
    error('Could not open fixture output');
end
cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id, '%s', jsonencode(fixture));
