import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
  Divider,
  Paper,
} from '@mui/material';
import {
  Memory,
  Speed,
  Storage,
  NetworkCheck,
  CheckCircle,
  Warning,
  Error,
  Info,
  Timeline,
  Straighten,
  Architecture,
  Polyline,
} from '@mui/icons-material';

interface SystemStatus {
  cpu: number;
  memory: number;
  storage: number;
  network: 'connected' | 'disconnected' | 'slow';
  solverStatus: 'ready' | 'running' | 'completed' | 'error';
  currentOperation: string;
  progress: number;
  meshQuality: number;
  elementCount: number;
  nodeCount: number;
}

const ProfessionalStatusBar: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus>({
    cpu: 23,
    memory: 67,
    storage: 45,
    network: 'connected',
    solverStatus: 'ready',
    currentOperation: 'Model Ready',
    progress: 0,
    meshQuality: 0.89,
    elementCount: 125847,
    nodeCount: 67293,
  });

  const [currentTime, setCurrentTime] = useState(new Date());
  const [showSystemDetails, setShowSystemDetails] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      // Simulate real-time updates
      setStatus(prev => ({
        ...prev,
        cpu: Math.max(10, Math.min(90, prev.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(30, Math.min(95, prev.memory + (Math.random() - 0.5) * 5)),
      }));
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  const handleSolverStatusClick = () => {
    console.log('Solver status clicked:', status.solverStatus);
    // TODO: Open solver configuration dialog
  };

  const handleModelStatsClick = (type: string) => {
    console.log(`Model ${type} clicked:`, type === 'nodes' ? status.nodeCount : type === 'elements' ? status.elementCount : status.meshQuality);
    // TODO: Open model statistics panel
  };

  const handleSystemResourceClick = (resource: string) => {
    console.log(`${resource} resource clicked`);
    setShowSystemDetails(!showSystemDetails);
    // TODO: Show detailed system resource monitoring
  };

  const handleNetworkClick = () => {
    console.log('Network status clicked:', status.network);
    // TODO: Open network diagnostics
  };

  const handleCoordinateSystemClick = () => {
    console.log('Coordinate system clicked');
    // TODO: Open coordinate system settings
  };

  const handleUnitsClick = () => {
    console.log('Units clicked');
    // TODO: Open unit system configuration
  };

  const getStatusIcon = (type: string) => {
    switch (status.solverStatus) {
      case 'ready':
        return <CheckCircle sx={{ fontSize: '14px', color: '#28a745' }} />;
      case 'running':
        return <Timeline sx={{ fontSize: '14px', color: '#ffc107' }} />;
      case 'completed':
        return <CheckCircle sx={{ fontSize: '14px', color: '#28a745' }} />;
      case 'error':
        return <Error sx={{ fontSize: '14px', color: '#dc3545' }} />;
      default:
        return <Info sx={{ fontSize: '14px', color: '#17a2b8' }} />;
    }
  };

  const getNetworkIcon = () => {
    switch (status.network) {
      case 'connected':
        return <NetworkCheck sx={{ fontSize: '14px', color: '#28a745' }} />;
      case 'slow':
        return <Warning sx={{ fontSize: '14px', color: '#ffc107' }} />;
      case 'disconnected':
        return <Error sx={{ fontSize: '14px', color: '#dc3545' }} />;
    }
  };

  return (
    <Paper
      elevation={0}
      sx={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: '#1e1e1e',
        borderTop: '1px solid #3e3e42',
        padding: '4px 12px',
        zIndex: 1300,
        minHeight: '28px',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, fontSize: '11px' }}>
        {/* Solver Status */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            cursor: 'pointer',
            padding: '2px 4px',
            borderRadius: '2px',
            '&:hover': { backgroundColor: '#3e3e42' }
          }}
          onClick={handleSolverStatusClick}
        >
          {getStatusIcon('solver')}
          <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
            {status.currentOperation}
          </Typography>
          {status.solverStatus === 'running' && (
            <LinearProgress
              variant="determinate"
              value={status.progress}
              sx={{
                width: '80px',
                height: '4px',
                backgroundColor: '#3e3e42',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: '#00a8ff',
                },
              }}
            />
          )}
        </Box>

        <Divider orientation="vertical" flexItem sx={{ backgroundColor: '#3e3e42' }} />

        {/* Model Statistics */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Tooltip title="Node Count - Click for details">
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                cursor: 'pointer',
                padding: '2px 4px',
                borderRadius: '2px',
                '&:hover': { backgroundColor: '#3e3e42' }
              }}
              onClick={() => handleModelStatsClick('nodes')}
            >
              <Polyline sx={{ fontSize: '12px', color: '#b0b0b0' }} />
              <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
                Nodes: {status.nodeCount.toLocaleString()}
              </Typography>
            </Box>
          </Tooltip>

          <Tooltip title="Element Count - Click for details">
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                cursor: 'pointer',
                padding: '2px 4px',
                borderRadius: '2px',
                '&:hover': { backgroundColor: '#3e3e42' }
              }}
              onClick={() => handleModelStatsClick('elements')}
            >
              <Architecture sx={{ fontSize: '12px', color: '#b0b0b0' }} />
              <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
                Elements: {status.elementCount.toLocaleString()}
              </Typography>
            </Box>
          </Tooltip>

          <Tooltip title="Mesh Quality - Click for analysis">
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                cursor: 'pointer',
                padding: '2px 4px',
                borderRadius: '2px',
                '&:hover': { backgroundColor: '#3e3e42' }
              }}
              onClick={() => handleModelStatsClick('quality')}
            >
              <Straighten sx={{ fontSize: '12px', color: '#b0b0b0' }} />
              <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
                Quality: {status.meshQuality.toFixed(2)}
              </Typography>
            </Box>
          </Tooltip>
        </Box>

        <Divider orientation="vertical" flexItem sx={{ backgroundColor: '#3e3e42' }} />

        {/* System Resources */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Tooltip title={`CPU Usage: ${status.cpu.toFixed(1)}% - Click for details`}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                cursor: 'pointer',
                padding: '2px 4px',
                borderRadius: '2px',
                '&:hover': { backgroundColor: '#3e3e42' }
              }}
              onClick={() => handleSystemResourceClick('CPU')}
            >
              <Speed sx={{ fontSize: '12px', color: '#b0b0b0' }} />
              <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
                CPU: {status.cpu.toFixed(0)}%
              </Typography>
            </Box>
          </Tooltip>

          <Tooltip title={`Memory Usage: ${status.memory.toFixed(1)}% - Click for details`}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                cursor: 'pointer',
                padding: '2px 4px',
                borderRadius: '2px',
                '&:hover': { backgroundColor: '#3e3e42' }
              }}
              onClick={() => handleSystemResourceClick('Memory')}
            >
              <Memory sx={{ fontSize: '12px', color: '#b0b0b0' }} />
              <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
                RAM: {status.memory.toFixed(0)}%
              </Typography>
            </Box>
          </Tooltip>

          <Tooltip title={`Storage Usage: ${status.storage}% - Click for details`}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                cursor: 'pointer',
                padding: '2px 4px',
                borderRadius: '2px',
                '&:hover': { backgroundColor: '#3e3e42' }
              }}
              onClick={() => handleSystemResourceClick('Storage')}
            >
              <Storage sx={{ fontSize: '12px', color: '#b0b0b0' }} />
              <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
                Disk: {status.storage}%
              </Typography>
            </Box>
          </Tooltip>
        </Box>

        <Divider orientation="vertical" flexItem sx={{ backgroundColor: '#3e3e42' }} />

        {/* Network Status */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            cursor: 'pointer',
            padding: '2px 4px',
            borderRadius: '2px',
            '&:hover': { backgroundColor: '#3e3e42' }
          }}
          onClick={handleNetworkClick}
        >
          {getNetworkIcon()}
          <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
            Network
          </Typography>
        </Box>

        <Box sx={{ flexGrow: 1 }} />

        {/* Coordinate System & Units */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            label="Global CS"
            size="small"
            clickable
            onClick={handleCoordinateSystemClick}
            sx={{
              height: '18px',
              fontSize: '10px',
              backgroundColor: '#2d2d30',
              color: '#b0b0b0',
              border: '1px solid #3e3e42',
              '&:hover': { backgroundColor: '#3e3e42' }
            }}
          />
          <Chip
            label="SI (m, kg, N, s, V, A)"
            size="small"
            clickable
            onClick={handleUnitsClick}
            sx={{
              height: '18px',
              fontSize: '10px',
              backgroundColor: '#2d2d30',
              color: '#b0b0b0',
              border: '1px solid #3e3e42',
              '&:hover': { backgroundColor: '#3e3e42' }
            }}
          />
        </Box>

        <Divider orientation="vertical" flexItem sx={{ backgroundColor: '#3e3e42' }} />

        {/* Time and Date */}
        <Typography variant="caption" sx={{ fontSize: '11px', color: '#b0b0b0' }}>
          {currentTime.toLocaleTimeString()} | {currentTime.toLocaleDateString()}
        </Typography>
      </Box>
    </Paper>
  );
};

export default ProfessionalStatusBar;