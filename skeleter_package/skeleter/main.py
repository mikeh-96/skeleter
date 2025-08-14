#!/usr/bin/env python3
"""
Skeleter - A command-line templating utility

This utility generates files based on input variables and secrets fetched from 
AWS Systems Manager Parameter Store. It uses Jinja2 templates and merges 
command-line variables with AWS Parameter Store values.

Usage:
    python skeleter.py --var key1=value1 --var key2=value2

Author: Generated for production use
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import urllib.parse

import boto3
import yaml
from botocore.exceptions import BotoCoreError, ClientError
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, UndefinedError
import git
from github import Github, GithubException
import requests
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter


class SkeletorError(Exception):
    """Base exception for Skeletor-related errors."""
    pass


class ConfigurationError(SkeletorError):
    """Raised when there are configuration-related issues."""
    pass


class ParameterStoreError(SkeletorError):
    """Raised when there are AWS Parameter Store-related issues."""
    pass


class TemplateError(SkeletorError):
    """Raised when there are template-related issues."""
    pass


class GitError(SkeletorError):
    """Raised when there are Git-related issues."""
    pass


class GitHubError(SkeletorError):
    """Raised when there are GitHub API-related issues."""
    pass


class Skeleter:
    """Main class for the Skeleter templating utility."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Skeleter with configuration file path.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.logger = self._setup_logging()
        self.ssm_client = None
        self.github_client = None
        self.temp_dir = None
        self.branch_name = f"skeleter-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.created_prs = []
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger('skeleter')
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
            self.logger.info(f"Loading configuration from {self.config_path}")
            with open(config_file, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            
            # Validate required sections
            required_sections = ['templates_map']
            for section in required_sections:
                if section not in self.config:
                    raise ConfigurationError(f"Missing required section '{section}' in config file")
            
            # Validate optional sections have correct types
            if 'parameter_store_map' in self.config and self.config['parameter_store_map'] is not None and not isinstance(self.config['parameter_store_map'], dict):
                raise ConfigurationError("'parameter_store_map' must be a dictionary")
            
            if 'input_variables' in self.config and self.config['input_variables'] is not None and not isinstance(self.config['input_variables'], dict):
                raise ConfigurationError("'input_variables' must be a dictionary")
            
            if 'github_config' in self.config and self.config['github_config'] is not None and not isinstance(self.config['github_config'], dict):
                raise ConfigurationError("'github_config' must be a dictionary")
            
            # Validate templates_map structure (repo -> {template: output} mapping)
            templates_map = self.config.get('templates_map', {})
            if templates_map and not isinstance(templates_map, dict):
                raise ConfigurationError("'templates_map' must be a dictionary")
            
            for repo_url, template_mapping in templates_map.items():
                if not isinstance(template_mapping, dict):
                    raise ConfigurationError(f"Template mapping for repository '{repo_url}' must be a dictionary")
            
            self.logger.info("Configuration loaded successfully")
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        except IOError as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")
    
    def _initialize_aws_client(self) -> None:
        """Initialize AWS Systems Manager client."""
        try:
            self.logger.info("Initializing AWS Systems Manager client")
            self.ssm_client = boto3.client('ssm')
            
            # Test the connection by making a simple call
            self.ssm_client.describe_parameters(MaxResults=1)
            self.logger.info("AWS client initialized successfully")
            
        except Exception as e:
            raise ParameterStoreError(f"Failed to initialize AWS client: {e}")
    
    def _initialize_github_client(self) -> None:
        """Initialize GitHub API client."""
        try:
            self.logger.info("Initializing GitHub API client")
            
            # Get GitHub token from config or environment variable
            github_config = self.config.get('github_config', {}) or {}
            token = github_config.get('token') or os.getenv('GITHUB_TOKEN')
            
            if not token:
                raise GitHubError("GitHub token not found. Set GITHUB_TOKEN environment variable or add token to github_config in config.yaml")
            
            self.github_client = Github(token)
            
            # Test the connection
            user = self.github_client.get_user()
            self.logger.info(f"GitHub client initialized successfully for user: {user.login}")
            
        except GithubException as e:
            raise GitHubError(f"Failed to initialize GitHub client: {e}")
        except Exception as e:
            raise GitHubError(f"Unexpected error initializing GitHub client: {e}")
    
    def fetch_parameters(self, context: Dict[str, str] = None) -> Dict[str, str]:
        """
        Fetch parameters from AWS Systems Manager Parameter Store.
        
        Args:
            context: Optional context for templating parameter paths
        
        Returns:
            Dictionary mapping parameter names to their values
        """
        parameters = {}
        parameter_store_map = self.config.get('parameter_store_map', {})
        
        # Handle None case (when commented out in YAML)
        if parameter_store_map is None:
            parameter_store_map = {}
        
        # If no parameters to fetch, return empty dict
        if not parameter_store_map:
            self.logger.info("No Parameter Store mappings configured, skipping AWS parameter fetch")
            return parameters
        
        if not self.ssm_client:
            self._initialize_aws_client()
        
        for param_name, param_path_template in parameter_store_map.items():
            try:
                # Render the parameter path if context is provided
                if context:
                    param_path = self._render_path_template(param_path_template, context)
                    self.logger.info(f"Rendered parameter path: {param_path_template} -> {param_path}")
                else:
                    param_path = param_path_template
                
                self.logger.info(f"Fetching parameter: {param_name} from {param_path}")
                
                response = self.ssm_client.get_parameter(
                    Name=param_path,
                    WithDecryption=True
                )
                
                parameters[param_name] = response['Parameter']['Value']
                self.logger.info(f"Successfully fetched parameter: {param_name}")
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ParameterNotFound':
                    raise ParameterStoreError(f"Parameter not found: {param_path}")
                elif error_code == 'AccessDenied':
                    raise ParameterStoreError(f"Access denied for parameter: {param_path}")
                else:
                    raise ParameterStoreError(f"AWS error fetching {param_path}: {e}")
            except TemplateError as e:
                self.logger.error(f"Failed to render parameter path template '{param_path_template}': {e}")
                continue
            except BotoCoreError as e:
                raise ParameterStoreError(f"AWS connection error fetching {param_path}: {e}")
        
        self.logger.info(f"Successfully fetched {len(parameters)} parameters from Parameter Store")
        return parameters
    
    def build_context(self, cli_vars: Dict[str, str]) -> Dict[str, str]:
        """
        Build the templating context by merging config variables, Parameter Store values, and CLI variables.
        
        Args:
            cli_vars: Variables passed from command line
            
        Returns:
            Merged context dictionary with precedence: CLI > Parameter Store > Config
        """
        self.logger.info("Building templating context")
        
        # Start with input variables from config (lowest precedence)
        context = {}
        config_inputs = self.config.get('input_variables', {})
        
        # Handle None case (when commented out in YAML)
        if config_inputs is None:
            config_inputs = {}
            
        if config_inputs:
            context.update(config_inputs)
            self.logger.info(f"Loaded {len(config_inputs)} variables from config: {list(config_inputs.keys())}")
        
        # Add CLI variables to create initial context for parameter path templating
        initial_context = context.copy()
        initial_context.update(cli_vars)
        
        # Fetch parameters from Parameter Store using the initial context (medium precedence)
        aws_parameters = self.fetch_parameters(initial_context)
        context.update(aws_parameters)
        
        # Add CLI variables (highest precedence)
        context.update(cli_vars)
        
        # Log context keys (but not values for security)
        context_keys = list(context.keys())
        self.logger.info(f"Context built with {len(context)} variables: {context_keys}")
        
        return context
    
    def process_templates(self, context: Dict[str, str]) -> None:
        """
        Process all repository templates using the provided context.
        
        Args:
            context: Dictionary containing template variables
        """
        templates_map = self.config.get('templates_map', {})
        
        if not templates_map:
            self.logger.warning("No repository templates defined in configuration")
            return
        
        self.logger.info(f"Processing templates in {len(templates_map)} repository(ies)")
        
        # Create temporary directory for Git operations
        self.temp_dir = tempfile.mkdtemp(prefix="skeleter-")
        self.logger.info(f"Created temporary directory: {self.temp_dir}")
        
        try:
            # Initialize GitHub client
            self._initialize_github_client()
            
            for repo_url, template_mappings in templates_map.items():
                try:
                    pr_url = self._process_repository(repo_url, template_mappings, context)
                    if pr_url:
                        self.created_prs.append(pr_url)
                        self.logger.info(f"Created PR: {pr_url}")
                except Exception as e:
                    self.logger.error(f"Failed to process repository {repo_url}: {e}")
                    raise TemplateError(f"Failed to process repository {repo_url}: {e}")
        
        finally:
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info("Cleaned up temporary directory")
    
    def _process_repository(self, repo_url: str, template_mappings: Dict[str, str], context: Dict[str, str]) -> str:
        """
        Process a single repository: clone, template, commit, and create PR.
        
        Args:
            repo_url: GitHub repository URL
            template_mappings: Mapping of template files to output paths
            context: Template variables
            
        Returns:
            Pull request URL if created successfully
        """
        self.logger.info(f"Processing repository: {repo_url}")
        
        # Parse repository information
        repo_info = self._parse_repository_url(repo_url)
        repo_owner, repo_name = repo_info['owner'], repo_info['name']
        
        # Clone repository
        repo_dir = os.path.join(self.temp_dir, f"{repo_owner}-{repo_name}")
        repo = self._clone_repository(repo_url, repo_dir)
        
        # Create and checkout feature branch
        self._create_feature_branch(repo, self.branch_name)
        
        # Process templates
        changes_made = False
        for template_path_template, output_path_template in template_mappings.items():
            # Render the template and output paths using context variables
            try:
                rendered_template_path = self._render_path_template(template_path_template, context)
                rendered_output_path = self._render_path_template(output_path_template, context)
                
                self.logger.info(f"Resolved paths: {template_path_template} -> {rendered_template_path}, {output_path_template} -> {rendered_output_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to render template paths: {e}")
                continue
            
            template_file = os.path.join(repo_dir, rendered_template_path)
            output_file = os.path.join(repo_dir, rendered_output_path)
            
            if not os.path.exists(template_file):
                self.logger.warning(f"Template file not found: {rendered_template_path} in {repo_url}")
                continue
            
            self._render_template_to_file(template_file, output_file, context)
            changes_made = True
        
        if not changes_made:
            self.logger.warning(f"No changes made to repository: {repo_url}")
            return None
        
        # Commit changes
        self._commit_changes(repo, context)
        
        # Push branch
        self._push_branch(repo, self.branch_name)
        
        # Create pull request
        pr_url = self._create_pull_request(repo_owner, repo_name, self.branch_name, context)
        
        return pr_url
    
    def _parse_repository_url(self, repo_url: str) -> Dict[str, str]:
        """Parse GitHub repository URL to extract owner and name."""
        try:
            # Handle both https://github.com/owner/repo and git@github.com:owner/repo.git formats
            if repo_url.startswith('https://github.com/'):
                path = repo_url.replace('https://github.com/', '').rstrip('.git')
            elif repo_url.startswith('git@github.com:'):
                path = repo_url.replace('git@github.com:', '').rstrip('.git')
            else:
                raise GitError(f"Unsupported repository URL format: {repo_url}")
            
            parts = path.split('/')
            if len(parts) != 2:
                raise GitError(f"Invalid repository URL format: {repo_url}")
            
            return {'owner': parts[0], 'name': parts[1]}
        
        except Exception as e:
            raise GitError(f"Failed to parse repository URL {repo_url}: {e}")
    
    def _clone_repository(self, repo_url: str, repo_dir: str) -> git.Repo:
        """Clone a Git repository."""
        try:
            self.logger.info(f"Cloning repository: {repo_url}")
            repo = git.Repo.clone_from(repo_url, repo_dir)
            self.logger.info(f"Repository cloned successfully to: {repo_dir}")
            return repo
        except git.GitCommandError as e:
            raise GitError(f"Failed to clone repository {repo_url}: {e}")
    
    def _create_feature_branch(self, repo: git.Repo, branch_name: str) -> None:
        """Create and checkout a new feature branch."""
        try:
            self.logger.info(f"Creating feature branch: {branch_name}")
            
            # Get default branch
            github_config = self.config.get('github_config', {}) or {}
            default_branch = github_config.get('target_branch', 'main')
            
            # Create and checkout new branch
            new_branch = repo.create_head(branch_name, f"origin/{default_branch}")
            new_branch.checkout()
            
            self.logger.info(f"Feature branch created and checked out: {branch_name}")
        
        except git.GitCommandError as e:
            raise GitError(f"Failed to create feature branch {branch_name}: {e}")
    
    def _commit_changes(self, repo: git.Repo, context: Dict[str, str]) -> None:
        """Commit changes to the repository."""
        try:
            # Add all changes
            repo.git.add(A=True)
            
            # Check if there are changes to commit
            if not repo.index.diff("HEAD"):
                self.logger.info("No changes to commit")
                return
            
            # Create commit message
            github_config = self.config.get('github_config', {}) or {}
            commit_message = f"{github_config.get('pr_title_prefix', 'Skeleter: Update configuration')}\n\nGenerated by Skeleter at {datetime.now().isoformat()}"
            
            # Commit changes
            repo.index.commit(commit_message)
            self.logger.info("Changes committed successfully")
        
        except git.GitCommandError as e:
            raise GitError(f"Failed to commit changes: {e}")
    
    def _push_branch(self, repo: git.Repo, branch_name: str) -> None:
        """Push the feature branch to remote."""
        try:
            self.logger.info(f"Pushing branch: {branch_name}")
            
            origin = repo.remote('origin')
            origin.push(branch_name)
            
            self.logger.info(f"Branch pushed successfully: {branch_name}")
        
        except git.GitCommandError as e:
            raise GitError(f"Failed to push branch {branch_name}: {e}")
    
    def _create_pull_request(self, repo_owner: str, repo_name: str, branch_name: str, context: Dict[str, str]) -> str:
        """Create a pull request on GitHub."""
        try:
            self.logger.info(f"Creating pull request for {repo_owner}/{repo_name}")
            
            # Get repository object
            repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
            
            # Get GitHub config
            github_config = self.config.get('github_config', {}) or {}
            target_branch = github_config.get('target_branch', 'main')
            title_prefix = github_config.get('pr_title_prefix', 'Skeleter: Update configuration')
            pr_body = github_config.get('pr_body', 'Automated configuration update generated by Skeleter')
            
            # Create PR title and body
            pr_title = f"{title_prefix} ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
            pr_description = f"{pr_body}\n\nBranch: {branch_name}\nGenerated: {datetime.now().isoformat()}"
            
            # Create pull request
            pr = repo.create_pull(
                title=pr_title,
                body=pr_description,
                head=branch_name,
                base=target_branch
            )
            
            pr_url = pr.html_url
            self.logger.info(f"Pull request created: {pr_url}")
            
            # Check if auto-merge is enabled from github_config
            github_config = self.config.get('github_config', {}) or {}
            force_merge = bool(github_config.get('force_merge', False))
            
            if force_merge:
                self.logger.info("Auto-merge enabled, attempting to merge PR")
                try:
                    pr.merge(merge_method='squash')
                    self.logger.info(f"Pull request merged automatically: {pr_url}")
                except GithubException as e:
                    self.logger.warning(f"Failed to auto-merge PR: {e}")
            
            return pr_url
        
        except GithubException as e:
            raise GitHubError(f"Failed to create pull request: {e}")
    
    def _render_template_to_file(self, template_path: str, output_path: str, context: Dict[str, str]) -> None:
        """
        Render a template file to an output file.
        
        Args:
            template_path: Path to the template file
            output_path: Path where the rendered output should be written
            context: Template variables
        """
        self.logger.info(f"Rendering template: {template_path} -> {output_path}")
        
        try:
            # Set up Jinja2 environment
            template_file = Path(template_path)
            template_dir = template_file.parent
            template_name = template_file.name
            
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Load and render template
            template = env.get_template(template_name)
            rendered_content = template.render(**context)
            
            # Format as YAML if the output file is a YAML file
            if self._is_yaml_file(output_path, template_path):
                self.logger.info(f"Formatting output as YAML: {output_path}")
                rendered_content = self._format_yaml_content(rendered_content)
            
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write rendered content
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(rendered_content)
            
            self.logger.info(f"Successfully rendered template to: {output_path}")
            
        except TemplateNotFound:
            raise TemplateError(f"Template not found: {template_path}")
        except UndefinedError as e:
            raise TemplateError(f"Undefined variable in template {template_path}: {e}")
        except IOError as e:
            raise TemplateError(f"Error writing output file {output_path}: {e}")
    
    def _render_template(self, template_path: str, output_path: str, context: Dict[str, str]) -> None:
        """
        Render a single template file.
        
        Args:
            template_path: Path to the template file
            output_path: Path where the rendered output should be written
            context: Template variables
        """
        self.logger.info(f"Rendering template: {template_path} -> {output_path}")
        
        # Validate template file exists
        template_file = Path(template_path)
        if not template_file.exists():
            raise TemplateError(f"Template file not found: {template_path}")
        
        try:
            # Set up Jinja2 environment
            template_dir = template_file.parent
            template_name = template_file.name
            
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Load and render template
            template = env.get_template(template_name)
            rendered_content = template.render(**context)
            
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write rendered content
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(rendered_content)
            
            self.logger.info(f"Successfully rendered template to: {output_path}")
            
        except TemplateNotFound:
            raise TemplateError(f"Template not found: {template_path}")
        except UndefinedError as e:
            raise TemplateError(f"Undefined variable in template {template_path}: {e}")
        except IOError as e:
            raise TemplateError(f"Error writing output file {output_path}: {e}")
    
    def _is_yaml_file(self, file_path: str, template_path: str = None) -> bool:
        """Check if a file should be treated as YAML based on its extension or template extension."""
        yaml_extensions = {'.yaml', '.yml'}
        
        # Check output file extension
        if Path(file_path).suffix.lower() in yaml_extensions:
            return True
        
        # Check template file extension if provided
        if template_path and Path(template_path).suffix.lower() in yaml_extensions:
            return True
        
        # Check for common YAML-related names
        file_name = Path(file_path).name.lower()
        if any(keyword in file_name for keyword in ['values', 'config', 'helm']):
            return True
        
        return False
    
    def _format_yaml_content(self, content: str) -> str:
        """Format content as properly indented YAML if it's valid YAML."""
        try:
            # First, try to clean up obvious formatting issues
            content = self._preprocess_yaml_content(content)
            
            # Try to parse as YAML with ruamel.yaml
            yaml_parser = YAML()
            yaml_parser.preserve_quotes = True
            yaml_parser.width = 4096  # Prevent line wrapping
            yaml_parser.indent(mapping=2, sequence=4, offset=2)
            yaml_parser.map_indent = 2
            yaml_parser.sequence_indent = 4
            
            # Parse the content
            data = yaml_parser.load(content)
            
            # If parsing succeeds, reformat it
            from io import StringIO
            output = StringIO()
            yaml_parser.dump(data, output)
            formatted_content = output.getvalue()
            
            self.logger.info("Successfully formatted content as YAML with ruamel.yaml")
            return formatted_content
            
        except Exception as e:
            self.logger.warning(f"YAML formatting with ruamel.yaml failed: {e}")
            # Fallback to manual formatting for common patterns
            try:
                formatted_content = self._manual_yaml_format(content)
                self.logger.info("Successfully formatted content using manual YAML formatting")
                return formatted_content
            except Exception as manual_error:
                self.logger.warning(f"Manual YAML formatting also failed: {manual_error}")
                return content
    
    def _preprocess_yaml_content(self, content: str) -> str:
        """Preprocess content to fix obvious YAML formatting issues."""
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                processed_lines.append(line)
                continue
            
            # Get indentation level
            current_indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            
            # Special handling for lines with multiple key:value pairs
            if ':' in stripped and '    ' in line and line.count(':') > 1:
                # This looks like multiple environment variables on one line
                # Pattern: "  env:    KEY1: value1    KEY2: value2    KEY3: value3"
                
                # Check if this starts with a key like "env:"
                if ':' in stripped and not stripped.startswith(' '):
                    parts = stripped.split(':', 1)
                    if len(parts) == 2:
                        main_key = parts[0].strip()
                        remaining = parts[1].strip()
                        
                        # Add the main key
                        processed_lines.append(' ' * current_indent + main_key + ':')
                        
                        # Process the remaining key:value pairs
                        env_pairs = self._split_env_pairs(remaining)
                        for key, value in env_pairs:
                            processed_lines.append(' ' * (current_indent + 2) + key + ': ' + value)
                        continue
            
            # Regular line processing
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _split_env_pairs(self, env_string: str) -> List[Tuple[str, str]]:
        """Split a string containing multiple KEY: value pairs."""
        pairs = []
        
        # Use regex to find KEY: value patterns
        import re
        
        # More sophisticated pattern to handle complex values including URIs
        # Look for WORD: followed by value (everything until next WORD: that's preceded by whitespace)
        pattern = r'(\w+):\s*([^:]*?(?::[^:\s]*?)*?)(?=\s+\w+:\s|$)'
        matches = re.findall(pattern, env_string)
        
        for key, value in matches:
            # Clean up the value
            value = value.strip()
            # Handle cases where the value might contain colons (like URLs)
            if value:
                pairs.append((key.strip(), value))
            else:
                pairs.append((key.strip(), ''))
        
        return pairs
    
    def _manual_yaml_format(self, content: str) -> str:
        """Manual YAML formatting as a fallback."""
        import re
        
        # Try to parse as YAML with standard PyYAML first
        try:
            data = yaml.safe_load(content)
            formatted = yaml.dump(data, default_flow_style=False, indent=2, sort_keys=False)
            return formatted
        except:
            pass
        
        # If that fails, try manual line-by-line processing
        lines = content.split('\n')
        formatted_lines = []
        current_indent = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Detect indentation level
            if stripped.endswith(':') and not stripped.startswith('-'):
                # This is a key
                formatted_lines.append('  ' * current_indent + stripped)
                current_indent += 1
            elif stripped.startswith('-'):
                # This is a list item
                formatted_lines.append('  ' * current_indent + stripped)
            elif ':' in stripped:
                # This is a key-value pair
                formatted_lines.append('  ' * current_indent + stripped)
            else:
                # Regular content
                formatted_lines.append('  ' * current_indent + stripped)
        
        return '\n'.join(formatted_lines)
    
    def _render_path_template(self, path_template: str, context: Dict[str, str]) -> str:
        """
        Render a path template using Jinja2 with the provided context.
        
        Args:
            path_template: Path template with Jinja2 variables (e.g., "helm/{{ service }}/values.yaml")
            context: Template variables
            
        Returns:
            Rendered path with variables substituted
        """
        try:
            # Create a simple Jinja2 environment for path rendering
            from jinja2 import Template
            
            template = Template(path_template)
            rendered_path = template.render(**context)
            
            # Validate the rendered path
            validated_path = self._validate_rendered_path(rendered_path)
            
            self.logger.debug(f"Rendered path template '{path_template}' -> '{validated_path}'")
            return validated_path
            
        except Exception as e:
            raise TemplateError(f"Failed to render path template '{path_template}': {e}")
    
    def _validate_rendered_path(self, path: str) -> str:
        """
        Validate a rendered path to ensure it's safe and valid.
        
        Args:
            path: The rendered path to validate
            
        Returns:
            The validated path
            
        Raises:
            TemplateError: If the path is invalid or unsafe
        """
        # Remove any leading/trailing whitespace
        path = path.strip()
        
        # Check for empty path
        if not path:
            raise TemplateError("Rendered path is empty")
        
        # Check for dangerous path traversal patterns
        # Note: For Parameter Store paths, leading '/' is valid and expected
        if '..' in path:
            raise TemplateError(f"Path contains unsafe patterns: {path}")
        
        # Check for invalid characters (basic validation)
        import re
        if re.search(r'[<>"|*?]', path):
            raise TemplateError(f"Path contains invalid characters: {path}")
        
        # Normalize path separators
        normalized_path = path.replace('\\', '/')
        
        return normalized_path
    
    def run(self, cli_vars: Dict[str, str], cli_force_merge: bool = False) -> None:
        """
        Main execution method.
        
        Args:
            cli_vars: Variables passed from command line
            cli_force_merge: Force merge flag from CLI
        """
        try:
            self.logger.info("Starting Skeleter execution")
            
            # Check if config_path needs to be resolved from variables
            final_config_path = self._resolve_config_path(cli_vars)
            if final_config_path != self.config_path:
                self.logger.info(f"Using config path from variable: {final_config_path}")
                self.config_path = final_config_path
            
            # Load configuration
            self.load_config()
            
            # Override force_merge in github_config if CLI flag is set
            if cli_force_merge:
                if 'github_config' not in self.config:
                    self.config['github_config'] = {}
                self.config['github_config']['force_merge'] = True
                self.logger.info("Force merge enabled via CLI flag")
            
            # Build context from CLI vars and Parameter Store
            context = self.build_context(cli_vars)
            
            # Process all templates
            self.process_templates(context)
            
            # Display created PRs
            if self.created_prs:
                print("\n🎉 Pull Requests Created:")
                for i, pr_url in enumerate(self.created_prs, 1):
                    print(f"  {i}. {pr_url}")
                print()
            else:
                print("ℹ️  No pull requests were created.")
            
            self.logger.info("Skeleter execution completed successfully")
            
        except SkeletorError as e:
            self.logger.error(f"Skeleter error: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            sys.exit(1)
    
    def _resolve_config_path(self, cli_vars: Dict[str, str]) -> str:
        """
        Resolve the final config path considering variable precedence.
        
        Args:
            cli_vars: Variables from command line
            
        Returns:
            Final config path to use
        """
        # If config_path is in CLI variables, check if we need to load a base config first
        if 'config_path' in cli_vars:
            return cli_vars['config_path']
        
        # Try to load base config to check for input_variables.config_path
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    base_config = yaml.safe_load(file) or {}
                    input_vars = base_config.get('input_variables', {}) or {}
                    if 'config_path' in input_vars:
                        return input_vars['config_path']
        except Exception as e:
            self.logger.debug(f"Could not check base config for config_path: {e}")
        
        # Return original path if no overrides found
        return self.config_path


def parse_arguments() -> Dict[str, str]:
    """
    Parse command-line arguments.
    
    Returns:
        Dictionary of key-value pairs from --var arguments
    """
    parser = argparse.ArgumentParser(
        description="Skeleter - Template generator with AWS Parameter Store integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python skeleter.py --var db_user=admin --var region=us-east-1
  python skeleter.py --var environment=production --var version=1.2.3
        """
    )
    
    parser.add_argument(
        '--var',
        action='append',
        dest='variables',
        metavar='KEY=VALUE',
        help='Set template variables (can be used multiple times)',
        default=[]
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--force-merge',
        action='store_true',
        help='Automatically merge pull requests without prompting'
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger('skeleter').setLevel(logging.DEBUG)
    
    # Parse key=value pairs
    variables = {}
    for var in args.variables:
        if '=' not in var:
            parser.error(f"Invalid variable format: {var}. Expected KEY=VALUE")
        
        key, value = var.split('=', 1)
        if not key.strip():
            parser.error(f"Empty key in variable: {var}")
        
        variables[key.strip()] = value
    
    # Note: force_merge CLI flag will be handled separately in the run method
    # It's not added to template variables since it's a GitHub config setting
    
    # Determine config path with precedence: CLI > variable > default
    config_path = args.config
    if 'config_path' in variables and args.config == 'config.yaml':  # Only override if CLI wasn't explicitly set
        config_path = variables['config_path']
        # Remove config_path from variables to avoid confusion in templates
        del variables['config_path']
    
    return variables, config_path, args.force_merge


def main():
    """Main entry point."""
    try:
        cli_vars, config_path, cli_force_merge = parse_arguments()
        
        skeleter = Skeleter(config_path)
        skeleter.run(cli_vars, cli_force_merge)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
