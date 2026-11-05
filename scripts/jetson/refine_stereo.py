"""Offline stereo candidate refinement from natural features; never installs a candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np


def load_stereo(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        intr = fs.getNode('camera_intrinsics')
        cameras = {role: {key: intr.getNode(role).getNode(key).mat() for key in ('K', 'D')}
                   for role in ('left', 'right')}
        node = fs.getNode('extrinsics').getNode('T_RL')
        return cameras, node.getNode('R').mat(), node.getNode('t').mat().ravel()
    finally:
        fs.release()


def match_pair(left_path, right_path, cameras):
    """Undistort before geometric filtering; distorted raw pixels need not share one F."""
    features = []
    detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=.025)
    for path in (left_path, right_path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (720, 1280):
            raise ValueError('Expected original 1280x720 stereo photos')
        features.append(detector.detectAndCompute(image, None))
    if any(desc is None for _, desc in features):
        raise ValueError('Insufficient stereo texture')
    pairs = cv2.BFMatcher().knnMatch(features[0][1], features[1][1], k=2)
    good = [m for m, n in pairs if m.distance < .7*n.distance]
    points = [np.array([features[side][0][getattr(m, index)].pt for m in good])
              for side, index in ((0, 'queryIdx'), (1, 'trainIdx'))]
    if len(good) < 100:
        raise ValueError('At least 100 unambiguous natural feature matches required')
    rays = [cv2.undistortPoints(p[:, None, :], cameras[role]['K'], cameras[role]['D'])
            .reshape(-1, 2) for role, p in zip(('left', 'right'), points)]
    _, mask = cv2.findFundamentalMat(rays[0]*500, rays[1]*500,
                                    cv2.USAC_MAGSAC, .7, .999, 10000)
    if mask is None or np.count_nonzero(mask) < 100:
        raise ValueError('Insufficient geometrically consistent stereo matches')
    return tuple(r[mask.ravel() > 0] for r in rays)


def pack_geometry(rotation, translation):
    t = translation/np.linalg.norm(translation)
    if t[0] >= -.9:
        raise ValueError('Requires a horizontal left-to-right negative-X baseline')
    return np.r_[cv2.Rodrigues(rotation)[0].ravel(),
                 np.arctan2(t[1], -t[0]), np.arctan2(t[2], np.hypot(t[0], t[1]))]


def unpack_geometry(parameters, baseline):
    theta, phi = parameters[3:]
    t = baseline*np.array([-np.cos(phi)*np.cos(theta),
                           np.cos(phi)*np.sin(theta), np.sin(phi)])
    return cv2.Rodrigues(parameters[:3])[0], t


def sampson_residual(parameters, left, right, focal=500):
    r, t = unpack_geometry(parameters, 1)
    cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    e = cross@r
    x, y = (np.column_stack((points, np.ones(len(points)))) for points in (left, right))
    ex, ety = (e@x.T).T, (e.T@y.T).T
    denominator = np.sqrt(ex[:, 0]**2+ex[:, 1]**2+ety[:, 0]**2+ety[:, 1]**2)
    return np.sum(y*ex, axis=1)/np.maximum(denominator, 1e-12)*focal


def fit_geometry(left, right, rotation, translation):
    from scipy.optimize import least_squares  # Offline optional dependency.

    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2 or len(left) < 100:
        raise ValueError('At least 100 paired normalized image points required')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Nonfinite image correspondences')
    initial = pack_geometry(rotation, translation)
    bounds = np.deg2rad([1, 1, 1, 3, 3])
    fit = least_squares(sampson_residual, initial, args=(left, right),
                        bounds=(initial-bounds, initial+bounds), loss='soft_l1',
                        f_scale=.3, max_nfev=400)
    result_r, result_t = unpack_geometry(fit.x, np.linalg.norm(translation))
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(result_r@rotation.T)[0])))
    if not fit.success or angle > .5 or np.any(np.abs(fit.x-initial) > bounds*.99):
        raise ValueError('Refinement is unconverged, excessive or at a search boundary')
    return result_r, result_t, {
        'matches': len(left), 'rotation_delta_deg': angle,
        'baseline_mm': float(np.linalg.norm(translation)),
        'original_sampson_px': np.percentile(
            np.abs(sampson_residual(initial, left, right)), [50, 90, 99]).tolist(),
        'candidate_sampson_px': np.percentile(
            np.abs(sampson_residual(fit.x, left, right)), [50, 90, 99]).tolist()}


def fit_rotation_with_plane(left, right, plane_points_right_m, plane_left,
                            rotation, translation, weight=30):
    """Metric plane constraints resolve yaw/scale ambiguity; keep all translation fixed."""
    from scipy.optimize import least_squares

    xyz = np.asarray(plane_points_right_m, np.float64)
    target = np.asarray(plane_left, np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 12
            or target.shape != (len(xyz), 2) or not np.isfinite(xyz).all()
            or not np.isfinite(target).all() or np.any(xyz[:, 2] < .2)
            or np.any(xyz[:, 2] > 10) or not 1 <= weight <= 30):
        raise ValueError('At least 12 valid metric plane points and normalized left pixels required')
    if left.shape != right.shape or left.shape[1:] != (2,) or len(left) < 100:
        raise ValueError('At least 100 paired scene correspondences required')
    initial = pack_geometry(rotation, translation)
    baseline = np.linalg.norm(translation)*.001

    def plane_error(parameters):
        r, t = unpack_geometry(parameters, baseline)
        points_left = (r.T@(xyz-t).T).T
        if np.any(points_left[:, 2] <= .1):
            raise ValueError('Plane projects behind the left camera')
        return (points_left[:, :2]/points_left[:, 2:]-target)*500

    def residual(rvec):
        parameters = np.r_[rvec, initial[3:]]
        return np.r_[sampson_residual(parameters, left, right),
                     plane_error(parameters).ravel()*weight]

    bounds = np.deg2rad([1, 1, 1])
    fit = least_squares(residual, initial[:3], bounds=(initial[:3]-bounds, initial[:3]+bounds),
                        loss='soft_l1', f_scale=.3, max_nfev=400)
    parameters = np.r_[fit.x, initial[3:]]
    result_r, result_t = unpack_geometry(parameters, np.linalg.norm(translation))
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(result_r@rotation.T)[0])))
    if not fit.success or angle > .5 or np.any(np.abs(fit.x-initial[:3]) > bounds*.99):
        raise ValueError('Metric rotation candidate is unconverged or excessive')
    return result_r, result_t, {
        'mode': 'rotation_only_metric_plane', 'matches': len(left),
        'metric_plane_points': len(xyz), 'plane_weight': weight,
        'rotation_delta_deg': angle, 'translation_fixed': True,
        'baseline_mm': float(np.linalg.norm(translation)),
        'original_plane_reprojection_px': np.percentile(np.abs(plane_error(initial)),
                                                       [50, 90], axis=0).tolist(),
        'candidate_plane_reprojection_px': np.percentile(np.abs(plane_error(parameters)),
                                                        [50, 90], axis=0).tolist()}


def write_candidate(source, destination, rotation, translation):
    cameras, _, old_t = load_stereo(source)
    if not np.isclose(np.linalg.norm(translation), np.linalg.norm(old_t), atol=1e-6):
        raise ValueError('Candidate must preserve the original metric baseline')
    fs = cv2.FileStorage(str(source), cv2.FILE_STORAGE_READ)
    try:
        p1 = fs.getNode('stereo_rectify').getNode('P1').mat()
    finally:
        fs.release()
    r1, r2, *_ = cv2.stereoRectify(
        cameras['left']['K'], cameras['left']['D'], cameras['right']['K'],
        cameras['right']['D'], (1280, 720), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    baseline = float(np.linalg.norm(translation))
    if not np.allclose(r2@translation, [-baseline, 0, 0], atol=1e-6):
        raise ValueError('Candidate is not horizontal rectified stereo')
    # Keep the original virtual intrinsics/FOV; only replace the rectification geometry.
    p2 = p1.copy()
    p2[0, 3] = -p1[0, 0]*baseline
    q = np.array([[1, 0, 0, -p1[0, 2]], [0, 1, 0, -p1[1, 2]],
                  [0, 0, 0, p1[0, 0]], [0, 0, 1/baseline, 0]])
    text = Path(source).read_text()

    def replace_matrix(block, name, value, indent):
        pattern = '('+re.escape(indent+name+': !!opencv-matrix')+r'\n.*?data: )\[.*?\]'
        values = '[ '+', '.join(format(float(v), '.17g') for v in value.ravel())+' ]'
        result, count = re.subn(pattern, lambda m: m.group(1)+values, block, count=1, flags=re.S)
        if count != 1:
            raise ValueError('Unsupported calibration serialization: '+name)
        return result

    start, end = text.index('   T_RL:'), text.index('   T_IR_L:')
    block = replace_matrix(text[start:end], 'R', rotation, '      ')
    block = replace_matrix(block, 't', translation, '      ')
    text = text[:start]+block+text[end:]
    for key, value in (('R1', r1), ('R2', r2), ('P1', p1), ('P2', p2), ('Q', q)):
        text = replace_matrix(text, key, value, '   ')
    text = text.replace('candidate_pending_physical_validation',
                        'candidate_epipolar_refinement_pending_validation')
    destination = Path(destination)
    if destination.resolve() == Path(source).resolve() or destination.exists():
        raise ValueError('Refusing to overwrite source or existing candidate')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def apply_rotation_override(source, destination, manifest):
    spec = json.loads(Path(manifest).read_text())
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if spec.get('status') in ('withdrawn', 'rejected'):
        raise ValueError('Rotation override has been withdrawn; do not reinstall')
    if (spec.get('schema_version') != 1 or spec.get('mode') != 'rotation_only_metric_plane'
            or spec.get('source_calibration_sha256') != digest):
        raise ValueError('Rotation override does not match the source calibration')
    _, old_r, old_t = load_stereo(source)
    r, t = np.array(spec['rotation'], np.float64), np.array(spec['translation_mm'], np.float64)
    if (r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all()
            or not np.isfinite(t).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-8)
            or not np.allclose(t, old_t, atol=1e-10, rtol=0)):
        raise ValueError('Override must be a proper rotation with unchanged translation')
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(r@old_r.T)[0])))
    if angle > .5:
        raise ValueError('Rotation override exceeds the validated correction range')
    result = write_candidate(source, destination, r, t)
    if result != spec.get('candidate_calibration_sha256'):
        Path(destination).unlink()  # Only the newly generated candidate; source is never replaced.
        raise ValueError('Rebuilt candidate hash does not match the validated artifact')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--override', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--metric-constraints', type=Path)
    parser.add_argument('--plane-weight', type=float, default=30)
    args = parser.parse_args()
    if args.override:
        digest = apply_rotation_override(args.calibration, args.out/'calibration_mm.yml',
                                         args.override)
        print(json.dumps({'candidate_sha256': digest, 'installed': False}))
        return
    if args.pairs is None:
        parser.error('--pairs is required when generating a new refinement')
    cameras, rotation, translation = load_stereo(args.calibration)
    files = sorted(p for p in args.pairs.glob('*_left.jpg') if p.stem.split('_')[0].isdigit())
    if len(files) < 3:
        raise ValueError('Three or more original stereo photo pairs required')
    matches = [match_pair(p, p.with_name(p.name.replace('_left', '_right')), cameras)
               for p in files]
    left, right = (np.concatenate([pair[index] for pair in matches]) for index in (0, 1))
    if args.metric_constraints:
        with np.load(args.metric_constraints, allow_pickle=False) as data:
            digest = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
            if str(data['source_sha256']) != digest:
                raise ValueError('Metric constraints belong to a different source calibration')
            r, t, report = fit_rotation_with_plane(data['left'], data['right'], data['plane3'],
                data['plane_left'], rotation, translation, args.plane_weight)
    else:
        r, t, report = fit_geometry(left, right, rotation, translation)
    holdout = match_pair(args.pairs/'holdout_left.jpg', args.pairs/'holdout_right.jpg', cameras)
    for name, rr, tt in (('original', rotation, translation), ('candidate', r, t)):
        report[name+'_holdout_sampson_px'] = np.percentile(np.abs(sampson_residual(
            pack_geometry(rr, tt), *holdout)), [50, 90, 99]).tolist()
    if report['candidate_holdout_sampson_px'][1] >= report['original_holdout_sampson_px'][1]*.8:
        raise ValueError('Candidate did not improve independent hold-out geometry by 20%')
    report.update(source_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                  candidate_sha256=write_candidate(args.calibration,
                      args.out/'calibration_mm.yml', r, t),
                  model_depth_validation_required=True, installed=False)
    (args.out/'refinement.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
