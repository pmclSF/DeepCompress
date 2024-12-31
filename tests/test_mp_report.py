import pytest
import os
import json
from tempfile import TemporaryDirectory
from mp_report import generate_report, load_experiment_results


# Helper function to create mock experiment results
def create_mock_experiment_results(output_dir):
    experiment_results = {
        'octree_levels': [3, 4],
        'quantization_levels': [8, 16],
        'timestamp': '2024-12-31 12:00:00',
        'original_1.ply': {
            'psnr': 35.0,
            'bd_rate': 0.25,
            'bitrate': 0.5,
            'compression_ratio': 0.75,
            'compression_time': 2.5,
            'decompression_time': 1.0
        },
        'original_2.ply': {
            'psnr': 30.0,
            'bd_rate': 0.30,
            'bitrate': 0.45,
            'compression_ratio': 0.80,
            'compression_time': 3.0,
            'decompression_time': 1.5
        },
        'original_3.ply': {
            'psnr': 40.0,
            'bd_rate': 0.20,
            'bitrate': 0.6,
            'compression_ratio': 0.85,
            'compression_time': 2.0,
            'decompression_time': 1.2
        }
    }

    # Save the experiment results to a temporary JSON file
    input_file = os.path.join(output_dir, 'experiment_results.json')
    with open(input_file, 'w') as f:
        json.dump(experiment_results, f, indent=4)
    
    return input_file, experiment_results


@pytest.fixture
def setup_experiment():
    """Fixture to set up experiment data."""
    with TemporaryDirectory() as temp_dir:
        input_file, experiment_results = create_mock_experiment_results(temp_dir)
        yield input_file, temp_dir, experiment_results


def test_generate_report(setup_experiment):
    """Test the generation of the experiment report."""
    input_file, output_dir, _ = setup_experiment

    # Generate the report using the mock experiment results
    output_file = os.path.join(output_dir, "experiment_report.json")
    generate_report(load_experiment_results(input_file), output_file)

    # Verify that the report is generated
    assert os.path.exists(output_file), "The report file was not created"

    # Check if the report contains the expected structure
    with open(output_file, 'r') as f:
        report = json.load(f)

    # Check if experiment metadata exists
    assert 'experiment_metadata' in report
    assert 'octree_levels' in report['experiment_metadata']
    assert 'quantization_levels' in report['experiment_metadata']

    # Check if the model performance section exists
    assert 'model_performance' in report
    assert len(report['model_performance']) == 3  # There should be 3 models

    # Check for best performance
    assert 'best_performance' in report
    assert 'best_psnr' in report['best_performance']
    assert 'best_bd_rate' in report['best_performance']


def test_best_performance_selection(setup_experiment):
    """Test that the best-performing model is correctly selected."""
    input_file, output_dir, experiment_results = setup_experiment

    # Generate the report
    output_file = os.path.join(output_dir, "experiment_report.json")
    generate_report(experiment_results, output_file)

    # Load the generated report
    with open(output_file, 'r') as f:
        report = json.load(f)

    # Check that the best performance is correctly identified
    best_performance = report['best_performance']

    # The best PSNR should be from "original_3.ply"
    assert best_performance['best_psnr'] == 'original_3.ply'
    # The best BD-Rate should be from "original_3.ply" (lowest BD-Rate is best)
    assert best_performance['best_bd_rate'] == 'original_3.ply'
    # The best bitrate should be from "original_2.ply" (lowest bitrate is best)
    assert best_performance['best_bitrate'] == 'original_2.ply'
    # The best compression ratio should be from "original_3.ply"
    assert best_performance['best_compression_ratio'] == 'original_3.ply'
    # The best compression time should be from "original_3.ply" (shorter is better)
    assert best_performance['best_compression_time'] == 'original_3.ply'
    # The best decompression time should be from "original_3.ply" (shorter is better)
    assert best_performance['best_decompression_time'] == 'original_3.ply'


def test_aggregate_statistics(setup_experiment):
    """Test that aggregate statistics like average PSNR and BD-Rate are computed correctly."""
    input_file, output_dir, experiment_results = setup_experiment

    # Generate the report
    output_file = os.path.join(output_dir, "experiment_report.json")
    generate_report(experiment_results, output_file)

    # Load the generated report
    with open(output_file, 'r') as f:
        report = json.load(f)

    # Check that the aggregate statistics section exists
    assert 'aggregate_statistics' in report

    # Ensure average PSNR and BD-Rate are computed
    assert 'avg_psnr' in report['aggregate_statistics']
    assert 'avg_bd_rate' in report['aggregate_statistics']
    assert isinstance(report['aggregate_statistics']['avg_psnr'], float)
    assert isinstance(report['aggregate_statistics']['avg_bd_rate'], float)

    # Check that the best model is correctly reported
    assert 'best_model' in report['aggregate_statistics']
    assert isinstance(report['aggregate_statistics']['best_model'], dict)


if __name__ == "__main__":
    pytest.main()
