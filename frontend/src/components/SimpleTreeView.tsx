import React, { useState } from 'react';
import {
  Box,
  Typography,
  Collapse,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  ExpandMore,
  ChevronRight,
  Folder,
  FolderOpen,
  Description,
  Architecture,
  ViewInAr,
  Science,
  Assessment,
} from '@mui/icons-material';

interface TreeNode {
  id: string;
  label: string;
  type: 'project' | 'geometry' | 'mesh' | 'physics' | 'solution' | 'results' | 'folder' | 'file';
  children?: TreeNode[];
  status?: 'ok' | 'warning' | 'error' | 'processing';
}

interface SimpleTreeViewProps {
  data: TreeNode[];
  onNodeSelect?: (nodeId: string) => void;
  selectedId?: string;
}

const SimpleTreeView: React.FC<SimpleTreeViewProps> = ({ data, onNodeSelect, selectedId }) => {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['project', 'geometry', 'mesh']));

  const getIcon = (type: string, isExpanded: boolean = false) => {
    switch (type) {
      case 'project': return <Description sx={{ fontSize: '16px', color: '#00a8ff' }} />;
      case 'geometry': return <ViewInAr sx={{ fontSize: '16px', color: '#28a745' }} />;
      case 'mesh': return <Architecture sx={{ fontSize: '16px', color: '#ffc107' }} />;
      case 'physics': return <Science sx={{ fontSize: '16px', color: '#dc3545' }} />;
      case 'results': return <Assessment sx={{ fontSize: '16px', color: '#17a2b8' }} />;
      case 'folder':
        return isExpanded ?
          <FolderOpen sx={{ fontSize: '16px', color: '#b0b0b0' }} /> :
          <Folder sx={{ fontSize: '16px', color: '#b0b0b0' }} />;
      default: return <Description sx={{ fontSize: '16px', color: '#b0b0b0' }} />;
    }
  };

  const handleToggle = (nodeId: string) => {
    const newExpanded = new Set(expanded);
    if (expanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpanded(newExpanded);
  };

  const handleSelect = (nodeId: string) => {
    if (onNodeSelect) {
      onNodeSelect(nodeId);
    }
  };

  const renderNode = (node: TreeNode, level: number = 0): React.ReactNode => {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expanded.has(node.id);
    const isSelected = selectedId === node.id;

    return (
      <Box key={node.id}>
        <ListItem disablePadding sx={{ pl: level * 2 }}>
          <ListItemButton
            onClick={() => handleSelect(node.id)}
            selected={isSelected}
            sx={{
              minHeight: '28px',
              padding: '2px 8px',
              '&:hover': {
                backgroundColor: '#3e3e42'
              },
              '&.Mui-selected': {
                backgroundColor: '#00a8ff',
                color: 'white',
                '&:hover': {
                  backgroundColor: '#0077cc'
                }
              }
            }}
          >
            {hasChildren && (
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  handleToggle(node.id);
                }}
                sx={{
                  padding: '2px',
                  marginRight: '4px',
                  color: 'inherit'
                }}
              >
                {isExpanded ? (
                  <ExpandMore sx={{ fontSize: '14px' }} />
                ) : (
                  <ChevronRight sx={{ fontSize: '14px' }} />
                )}
              </IconButton>
            )}
            <ListItemIcon sx={{ minWidth: '24px', marginRight: '8px' }}>
              {getIcon(node.type, isExpanded)}
            </ListItemIcon>
            <ListItemText
              primary={node.label}
              primaryTypographyProps={{
                fontSize: '12px',
                color: 'inherit'
              }}
            />
          </ListItemButton>
        </ListItem>

        {hasChildren && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {node.children!.map((child) => renderNode(child, level + 1))}
            </List>
          </Collapse>
        )}
      </Box>
    );
  };

  return (
    <List component="nav" disablePadding>
      {data.map((node) => renderNode(node))}
    </List>
  );
};

export default SimpleTreeView;