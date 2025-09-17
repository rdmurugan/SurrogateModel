import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Chip,
} from '@mui/material';
import SimpleTreeView from './SimpleTreeView';
import GeometryImport from './dialogs/GeometryImport';
import NodeProperties from './dialogs/NodeProperties';
import {
  ExpandMore,
  Add,
  Delete,
  Edit,
  FileCopy,
  Refresh,
  Visibility,
  CloudUpload,
  Engineering,
  Science,
  VisibilityOff,
  PlayArrow
} from '@mui/icons-material';

interface TreeNode {
  id: string;
  label: string;
  type: 'project' | 'geometry' | 'mesh' | 'physics' | 'solution' | 'results' | 'folder' | 'file';
  children?: TreeNode[];
  visible?: boolean;
  status?: 'ok' | 'warning' | 'error' | 'processing';
  properties?: Record<string, any>;
}

// Define the initial model tree first
const initialModelTree: TreeNode = {
  id: 'project',
  label: 'Structural Analysis Project',
  type: 'project',
  status: 'ok',
  children: [
    {
      id: 'geometry',
      label: 'Geometry',
      type: 'geometry',
      status: 'ok',
      children: [
        { id: 'part1', label: 'Main Assembly', type: 'file', status: 'ok' },
        { id: 'part2', label: 'Support Bracket', type: 'file', status: 'ok' },
        { id: 'part3', label: 'Load Plate', type: 'file', status: 'warning' },
      ]
    },
    {
      id: 'mesh',
      label: 'Mesh',
      type: 'mesh',
      status: 'ok',
      children: [
        { id: 'mesh1', label: 'Body Sizing', type: 'file', status: 'ok' },
        { id: 'mesh2', label: 'Face Sizing', type: 'file', status: 'ok' },
        { id: 'mesh3', label: 'Edge Sizing', type: 'file', status: 'processing' },
      ]
    },
    {
      id: 'physics',
      label: 'Physics Setup',
      type: 'physics',
      status: 'ok',
      children: [
        {
          id: 'materials',
          label: 'Materials',
          type: 'folder',
          children: [
            { id: 'steel', label: 'Structural Steel', type: 'file', status: 'ok' },
            { id: 'aluminum', label: 'Aluminum Alloy', type: 'file', status: 'ok' },
          ]
        },
        {
          id: 'loads',
          label: 'Loads and Constraints',
          type: 'folder',
          children: [
            { id: 'force1', label: 'Applied Force (1000 N)', type: 'file', status: 'ok' },
            { id: 'constraint1', label: 'Fixed Support', type: 'file', status: 'ok' },
            { id: 'pressure1', label: 'Surface Pressure', type: 'file', status: 'warning' },
          ]
        }
      ]
    },
    {
      id: 'solution',
      label: 'Solution',
      type: 'solution',
      status: 'processing',
      children: [
        { id: 'solver1', label: 'Static Structural', type: 'file', status: 'processing' },
        { id: 'solver2', label: 'Modal Analysis', type: 'file', status: 'ok' },
      ]
    },
    {
      id: 'results',
      label: 'Results',
      type: 'results',
      status: 'ok',
      children: [
        { id: 'stress', label: 'Total Deformation', type: 'file', status: 'ok' },
        { id: 'strain', label: 'Equivalent Stress', type: 'file', status: 'ok' },
        { id: 'safety', label: 'Safety Factor', type: 'file', status: 'ok' },
        { id: 'modes', label: 'Mode Shapes', type: 'file', status: 'warning' },
      ]
    }
  ]
};

// Helper functions for tree operations
const findNodeById = (node: TreeNode, id: string): TreeNode | null => {
  if (node.id === id) return node;
  if (node.children) {
    for (const child of node.children) {
      const found = findNodeById(child, id);
      if (found) return found;
    }
  }
  return null;
};

const updateNodeVisibility = (node: TreeNode, id: string): TreeNode => {
  if (node.id === id) {
    return { ...node, visible: !node.visible };
  }
  if (node.children) {
    return {
      ...node,
      children: node.children.map(child => updateNodeVisibility(child, id))
    };
  }
  return node;
};

const deleteNode = (node: TreeNode, id: string): TreeNode => {
  if (node.children) {
    const filteredChildren = node.children.filter(child => child.id !== id);
    return {
      ...node,
      children: filteredChildren.map(child => deleteNode(child, id))
    };
  }
  return node;
};

const getAllNodeIds = (node: TreeNode): string[] => {
  const ids = [node.id];
  if (node.children) {
    node.children.forEach(child => {
      ids.push(...getAllNodeIds(child));
    });
  }
  return ids;
};

const ModelTreePanel: React.FC = () => {
  const [selected, setSelected] = useState<string[]>(['project']);
  const [contextMenu, setContextMenu] = useState<{
    mouseX: number;
    mouseY: number;
    nodeId: string;
  } | null>(null);
  const [geometryImportOpen, setGeometryImportOpen] = useState(false);
  const [nodePropertiesOpen, setNodePropertiesOpen] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNodeType, setSelectedNodeType] = useState<string>('project');
  const [modelTree, setModelTree] = useState<TreeNode>(initialModelTree);
  const [expandedNodes, setExpandedNodes] = useState<string[]>(['project', 'geometry', 'mesh', 'physics']);

  const handleContextMenu = (event: React.MouseEvent, nodeId: string) => {
    event.preventDefault();
    setContextMenu(
      contextMenu === null
        ? { mouseX: event.clientX - 2, mouseY: event.clientY - 4, nodeId }
        : null
    );
  };

  const handleCloseContextMenu = () => {
    setContextMenu(null);
  };

  const handleEditNode = () => {
    if (contextMenu) {
      const node = findNodeById(modelTree, contextMenu.nodeId);
      if (node) {
        setSelectedNodeId(contextMenu.nodeId);
        setSelectedNodeType(node.type);
        setNodePropertiesOpen(true);
      }
    }
    handleCloseContextMenu();
  };

  const handleDuplicateNode = () => {
    if (contextMenu) {
      console.log('Duplicate node:', contextMenu.nodeId);
      // TODO: Implement node duplication
    }
    handleCloseContextMenu();
  };

  const handleToggleVisibility = () => {
    if (contextMenu) {
      setModelTree(prev => updateNodeVisibility(prev, contextMenu.nodeId));
    }
    handleCloseContextMenu();
  };

  const handleDeleteNode = () => {
    if (contextMenu) {
      if (window.confirm(`Are you sure you want to delete this item?`)) {
        setModelTree(prev => deleteNode(prev, contextMenu.nodeId));
      }
    }
    handleCloseContextMenu();
  };

  const handleRefreshTree = () => {
    setModelTree(initialModelTree);
    setExpandedNodes(['project', 'geometry', 'mesh', 'physics']);
  };

  const handleAddGeometry = () => {
    setGeometryImportOpen(true);
  };

  const handleImportGeometry = (files: any[]) => {
    // Add imported files to the geometry section
    const newGeometryNodes = files.map(file => ({
      id: `geo_${file.id}`,
      label: file.name,
      type: 'file' as const,
      status: 'ok' as const,
      visible: file.visible
    }));

    setModelTree(prev => {
      const updatedTree = { ...prev };
      if (updatedTree.children) {
        const geometryNode = updatedTree.children.find(n => n.id === 'geometry');
        if (geometryNode && geometryNode.children) {
          geometryNode.children.push(...newGeometryNodes);
        }
      }
      return updatedTree;
    });
  };

  const handleRunAnalysis = () => {
    // Update solution status to processing
    setModelTree(prev => {
      const updatedTree = { ...prev };
      if (updatedTree.children) {
        const solutionNode = updatedTree.children.find(n => n.id === 'solution');
        if (solutionNode) {
          solutionNode.status = 'processing';
        }
      }
      return updatedTree;
    });

    // Simulate analysis completion after 5 seconds
    setTimeout(() => {
      setModelTree(prev => {
        const updatedTree = { ...prev };
        if (updatedTree.children) {
          const solutionNode = updatedTree.children.find(n => n.id === 'solution');
          if (solutionNode) {
            solutionNode.status = 'ok';
          }
        }
        return updatedTree;
      });
    }, 5000);
  };

  const handleNodePropertiesUpdate = (nodeId: string, properties: any) => {
    console.log(`Updating node ${nodeId} with properties:`, properties);
    // Update the node with new properties
    setModelTree(prev => {
      // This would update the tree with new properties
      // For now just log the update
      return prev;
    });
  };

  const handleExpandAll = () => {
    const allNodeIds = getAllNodeIds(modelTree);
    setExpandedNodes(allNodeIds);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Panel Header */}
      <Paper
        elevation={0}
        sx={{
          backgroundColor: '#2d2d30',
          borderBottom: '1px solid #3e3e42',
          padding: '6px 12px',
          minHeight: '32px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        <Typography variant="subtitle2" sx={{ fontSize: '12px', fontWeight: 600 }}>
          Model Tree
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          <Tooltip title="Import Geometry">
            <IconButton size="small" sx={{ padding: '2px' }} onClick={handleAddGeometry}>
              <CloudUpload sx={{ fontSize: '14px' }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Run Analysis">
            <IconButton size="small" sx={{ padding: '2px' }} onClick={handleRunAnalysis}>
              <PlayArrow sx={{ fontSize: '14px' }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh Model Tree">
            <IconButton size="small" sx={{ padding: '2px' }} onClick={handleRefreshTree}>
              <Refresh sx={{ fontSize: '14px' }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Expand All Nodes">
            <IconButton size="small" sx={{ padding: '2px' }} onClick={handleExpandAll}>
              <Add sx={{ fontSize: '14px' }} />
            </IconButton>
          </Tooltip>
        </Box>
      </Paper>

      {/* Tree View */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', padding: '8px' }}>
        <SimpleTreeView
          data={[modelTree]}
          onNodeSelect={(nodeId) => setSelected([nodeId])}
          selectedId={selected[0]}
        />
      </Box>

      {/* Properties Panel */}
      <Paper
        elevation={0}
        sx={{
          backgroundColor: '#2d2d30',
          borderTop: '1px solid #3e3e42',
          maxHeight: '300px',
          overflow: 'auto'
        }}
      >
        <Accordion defaultExpanded>
          <AccordionSummary
            expandIcon={<ExpandMore sx={{ fontSize: '14px' }} />}
            sx={{
              minHeight: '32px !important',
              '& .MuiAccordionSummary-content': {
                margin: '6px 0 !important'
              }
            }}
          >
            <Typography variant="subtitle2" sx={{ fontSize: '12px', fontWeight: 600 }}>
              Properties
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ padding: '8px 16px' }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <TextField
                label="Name"
                defaultValue="Structural Analysis Project"
                size="small"
                variant="outlined"
                InputProps={{ style: { fontSize: '11px' } }}
                InputLabelProps={{ style: { fontSize: '11px' } }}
              />
              <TextField
                label="Analysis Type"
                defaultValue="Static Structural"
                size="small"
                variant="outlined"
                InputProps={{ style: { fontSize: '11px' } }}
                InputLabelProps={{ style: { fontSize: '11px' } }}
              />
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                <Chip
                  label="Active"
                  size="small"
                  color="success"
                  sx={{ fontSize: '10px', height: '20px' }}
                />
                <Chip
                  label="Modified"
                  size="small"
                  color="warning"
                  sx={{ fontSize: '10px', height: '20px' }}
                />
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        <Accordion>
          <AccordionSummary
            expandIcon={<ExpandMore sx={{ fontSize: '14px' }} />}
            sx={{
              minHeight: '32px !important',
              '& .MuiAccordionSummary-content': {
                margin: '6px 0 !important'
              }
            }}
          >
            <Typography variant="subtitle2" sx={{ fontSize: '12px', fontWeight: 600 }}>
              Statistics
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ padding: '8px 16px' }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ fontSize: '11px' }}>Nodes:</Typography>
                <Typography variant="caption" sx={{ fontSize: '11px', fontWeight: 600 }}>67,293</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ fontSize: '11px' }}>Elements:</Typography>
                <Typography variant="caption" sx={{ fontSize: '11px', fontWeight: 600 }}>125,847</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ fontSize: '11px' }}>Bodies:</Typography>
                <Typography variant="caption" sx={{ fontSize: '11px', fontWeight: 600 }}>3</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ fontSize: '11px' }}>Materials:</Typography>
                <Typography variant="caption" sx={{ fontSize: '11px', fontWeight: 600 }}>2</Typography>
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>
      </Paper>

      {/* Context Menu */}
      <Menu
        open={contextMenu !== null}
        onClose={handleCloseContextMenu}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
        PaperProps={{
          sx: {
            backgroundColor: '#2d2d30',
            border: '1px solid #3e3e42',
            fontSize: '12px'
          }
        }}
      >
        <MenuItem onClick={handleEditNode} sx={{ fontSize: '12px' }}>
          <Edit sx={{ mr: 1, fontSize: '14px' }} />
          Edit Properties
        </MenuItem>
        <MenuItem onClick={handleDuplicateNode} sx={{ fontSize: '12px' }}>
          <FileCopy sx={{ mr: 1, fontSize: '14px' }} />
          Duplicate
        </MenuItem>
        <MenuItem onClick={handleToggleVisibility} sx={{ fontSize: '12px' }}>
          <Visibility sx={{ mr: 1, fontSize: '14px' }} />
          Toggle Visibility
        </MenuItem>
        <Divider sx={{ backgroundColor: '#3e3e42' }} />
        <MenuItem onClick={handleDeleteNode} sx={{ fontSize: '12px', color: '#dc3545' }}>
          <Delete sx={{ mr: 1, fontSize: '14px' }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Dialogs */}
      <GeometryImport
        open={geometryImportOpen}
        onClose={() => setGeometryImportOpen(false)}
        onImport={handleImportGeometry}
      />

      <NodeProperties
        open={nodePropertiesOpen}
        onClose={() => setNodePropertiesOpen(false)}
        nodeId={selectedNodeId}
        nodeType={selectedNodeType}
        onUpdate={handleNodePropertiesUpdate}
      />

      <style>
        {`
          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
          }
        `}
      </style>
    </Box>
  );
};

export default ModelTreePanel;