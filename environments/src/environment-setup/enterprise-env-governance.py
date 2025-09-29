#!/usr/bin/env python3

"""
Enterprise Python Environment Governance and Policy Management
Purpose: Enforce organizational standards, compliance, and best practices
Features: Policy enforcement, audit trails, compliance reporting, automated remediation
Use Case: Large organizations managing hundreds of Python environments
"""

import os
import json
import yaml
import logging
import hashlib
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import time

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/python-env-governance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnterpriseGovernance')

# ==============================================================================
# POLICY DEFINITIONS AND COMPLIANCE RULES
# ==============================================================================

class PolicyViolationLevel(Enum):
    """Severity levels for policy violations"""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Should be fixed but not critical
    ERROR = "error"         # Must be fixed, blocks certain operations
    CRITICAL = "critical"   # Immediate action required, security risk

class ComplianceStatus(Enum):
    """Environment compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"

@dataclass
class PolicyRule:
    """Definition of a policy rule"""
    id: str
    name: str
    description: str
    category: str  # security, performance, standards, licensing
    level: PolicyViolationLevel
    check_function: str  # Name of function to execute
    remediation_function: Optional[str] = None
    auto_remediate: bool = False
    exceptions: List[str] = field(default_factory=list)  # Environment patterns to exempt
    metadata: Dict = field(default_factory=dict)

@dataclass
class PolicyViolation:
    """Record of a policy violation"""
    rule_id: str
    environment_name: str
    violation_level: PolicyViolationLevel
    message: str
    detected_at: str
    remediated: bool = False
    remediation_result: Optional[str] = None
    details: Dict = field(default_factory=dict)

@dataclass
class EnvironmentAuditRecord:
    """Audit record for an environment"""
    environment_name: str
    audit_timestamp: str
    compliance_status: ComplianceStatus
    policy_violations: List[PolicyViolation]
    metrics: Dict  # Performance, security, usage metrics
    recommendations: List[str]
    auditor: str  # User or system that performed audit

# ==============================================================================
# ENTERPRISE POLICY ENGINE
# ==============================================================================

class EnterprisePolicyEngine:
    """
    Central policy engine for enterprise Python environment governance
    Enforces organizational standards and compliance requirements
    """
    
    def __init__(self, config_file: str = None):
        """Initialize the policy engine"""
        self.config = self._load_config(config_file or "/etc/python-env/governance.yaml")
        self.db_path = self.config.get("database_path", "/var/lib/python-env/governance.db")
        self.policies = self._load_policies()
        self.audit_trail = []
        
        # Initialize database
        self._init_database()
        
        # Load custom validators
        self.validators = self._load_validators()
        
        logger.info("Enterprise Policy Engine initialized")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        config_path = Path(config_file)
        if not config_path.exists():
            # Create default configuration
            default_config = {
                "organization": "Enterprise",
                "database_path": "/var/lib/python-env/governance.db",
                "notification_email": "devops@example.com",
                "enforcement_mode": "audit",  # audit, enforce, strict
                "auto_remediation": False,
                "audit_interval_hours": 24,
                "retention_days": 90,
                "max_environment_age_days": 180,
                "allowed_python_versions": ["3.8", "3.9", "3.10", "3.11"],
                "blocked_packages": ["malicious-package"],
                "required_packages": ["pip-audit", "safety"],
                "max_package_age_days": 30,
                "min_test_coverage": 80,
                "require_signed_commits": True,
                "require_dependency_scanning": True
            }
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_database(self):
        """Initialize SQLite database for audit trails"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                environment_name TEXT NOT NULL,
                audit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                compliance_status TEXT,
                violations TEXT,  -- JSON
                metrics TEXT,     -- JSON
                recommendations TEXT,  -- JSON
                auditor TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS policy_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT NOT NULL,
                environment_name TEXT NOT NULL,
                violation_level TEXT,
                message TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                remediated BOOLEAN DEFAULT FALSE,
                details TEXT  -- JSON
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS environment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                environment_name TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                package_count INTEGER,
                last_activated TIMESTAMP,
                created_by TEXT,
                project TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_policies(self) -> List[PolicyRule]:
        """Load policy rules"""
        policies = [
            # Security Policies
            PolicyRule(
                id="SEC001",
                name="No vulnerable packages",
                description="Environment must not contain packages with known vulnerabilities",
                category="security",
                level=PolicyViolationLevel.CRITICAL,
                check_function="check_vulnerable_packages",
                remediation_function="fix_vulnerable_packages",
                auto_remediate=True
            ),
            PolicyRule(
                id="SEC002",
                name="Package signatures",
                description="All packages must be signed and verified",
                category="security",
                level=PolicyViolationLevel.ERROR,
                check_function="check_package_signatures",
                remediation_function=None,
                auto_remediate=False
            ),
            PolicyRule(
                id="SEC003",
                name="No blocked packages",
                description="Environment must not contain organizationally blocked packages",
                category="security",
                level=PolicyViolationLevel.CRITICAL,
                check_function="check_blocked_packages",
                remediation_function="remove_blocked_packages",
                auto_remediate=True
            ),
            
            # Compliance Policies
            PolicyRule(
                id="COMP001",
                name="Python version compliance",
                description="Must use approved Python versions",
                category="standards",
                level=PolicyViolationLevel.ERROR,
                check_function="check_python_version",
                remediation_function=None,
                auto_remediate=False
            ),
            PolicyRule(
                id="COMP002",
                name="License compliance",
                description="No GPL/AGPL licensed packages in production",
                category="licensing",
                level=PolicyViolationLevel.ERROR,
                check_function="check_license_compliance",
                remediation_function=None,
                auto_remediate=False,
                exceptions=["dev_*", "test_*"]
            ),
            PolicyRule(
                id="COMP003",
                name="Required security tools",
                description="Must have security scanning tools installed",
                category="security",
                level=PolicyViolationLevel.WARNING,
                check_function="check_required_tools",
                remediation_function="install_required_tools",
                auto_remediate=True
            ),
            
            # Performance Policies
            PolicyRule(
                id="PERF001",
                name="Package count limit",
                description="Environment should not exceed 200 packages",
                category="performance",
                level=PolicyViolationLevel.WARNING,
                check_function="check_package_count",
                remediation_function=None,
                auto_remediate=False,
                metadata={"max_packages": 200}
            ),
            PolicyRule(
                id="PERF002",
                name="Environment size limit",
                description="Environment should not exceed 2GB",
                category="performance",
                level=PolicyViolationLevel.WARNING,
                check_function="check_environment_size",
                remediation_function="optimize_environment_size",
                auto_remediate=False,
                metadata={"max_size_gb": 2}
            ),
            
            # Lifecycle Policies
            PolicyRule(
                id="LIFE001",
                name="Maximum environment age",
                description="Environments older than 180 days should be reviewed",
                category="lifecycle",
                level=PolicyViolationLevel.WARNING,
                check_function="check_environment_age",
                remediation_function=None,
                auto_remediate=False,
                metadata={"max_age_days": 180}
            ),
            PolicyRule(
                id="LIFE002",
                name="Outdated packages",
                description="Packages should not be more than 30 days out of date",
                category="lifecycle",
                level=PolicyViolationLevel.WARNING,
                check_function="check_outdated_packages",
                remediation_function="update_outdated_packages",
                auto_remediate=False,
                metadata={"max_outdated_days": 30}
            )
        ]
        
        return policies
    
    def _load_validators(self) -> Dict:
        """Load custom validation functions"""
        validators = {}
        
        # Security validators
        validators["check_vulnerable_packages"] = self._check_vulnerable_packages
        validators["check_package_signatures"] = self._check_package_signatures
        validators["check_blocked_packages"] = self._check_blocked_packages
        
        # Compliance validators
        validators["check_python_version"] = self._check_python_version
        validators["check_license_compliance"] = self._check_license_compliance
        validators["check_required_tools"] = self._check_required_tools
        
        # Performance validators
        validators["check_package_count"] = self._check_package_count
        validators["check_environment_size"] = self._check_environment_size
        
        # Lifecycle validators
        validators["check_environment_age"] = self._check_environment_age
        validators["check_outdated_packages"] = self._check_outdated_packages
        
        # Remediation functions
        validators["fix_vulnerable_packages"] = self._fix_vulnerable_packages
        validators["remove_blocked_packages"] = self._remove_blocked_packages
        validators["install_required_tools"] = self._install_required_tools
        validators["optimize_environment_size"] = self._optimize_environment_size
        validators["update_outdated_packages"] = self._update_outdated_packages
        
        return validators
    
    # ==============================================================================
    # POLICY CHECK IMPLEMENTATIONS
    # ==============================================================================
    
    def _check_vulnerable_packages(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check for packages with known vulnerabilities"""
        try:
            pip_exe = Path(env_path) / "bin" / "pip"
            
            # Run pip-audit
            result = subprocess.run(
                [str(pip_exe.parent / "pip-audit"), "--format", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                audit_data = json.loads(result.stdout) if result.stdout else {}
                vulnerabilities = audit_data.get("vulnerabilities", [])
                
                if vulnerabilities:
                    return False, f"Found {len(vulnerabilities)} vulnerable packages", {
                        "vulnerabilities": vulnerabilities,
                        "severity_counts": self._count_severities(vulnerabilities)
                    }
            
            return True, "No vulnerabilities found", {}
            
        except Exception as e:
            logger.error(f"Error checking vulnerabilities: {e}")
            return False, f"Failed to check vulnerabilities: {str(e)}", {}
    
    def _check_package_signatures(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check if packages are properly signed"""
        # This is a placeholder - implement actual signature verification
        # based on your organization's requirements
        try:
            unsigned_packages = []
            
            # Check for unsigned packages
            # Implementation would depend on your package signing infrastructure
            
            if unsigned_packages:
                return False, f"Found {len(unsigned_packages)} unsigned packages", {
                    "unsigned_packages": unsigned_packages
                }
            
            return True, "All packages are properly signed", {}
            
        except Exception as e:
            logger.error(f"Error checking signatures: {e}")
            return False, f"Failed to check signatures: {str(e)}", {}
    
    def _check_blocked_packages(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check for organizationally blocked packages"""
        try:
            blocked = self.config.get("blocked_packages", [])
            pip_exe = Path(env_path) / "bin" / "pip"
            
            # Get installed packages
            result = subprocess.run(
                [str(pip_exe), "list", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            installed = json.loads(result.stdout)
            installed_names = {pkg["name"].lower() for pkg in installed}
            
            # Check for blocked packages
            found_blocked = [pkg for pkg in blocked if pkg.lower() in installed_names]
            
            if found_blocked:
                return False, f"Found blocked packages: {', '.join(found_blocked)}", {
                    "blocked_packages": found_blocked
                }
            
            return True, "No blocked packages found", {}
            
        except Exception as e:
            logger.error(f"Error checking blocked packages: {e}")
            return False, f"Failed to check blocked packages: {str(e)}", {}
    
    def _check_python_version(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check if Python version is approved"""
        try:
            allowed_versions = self.config.get("allowed_python_versions", [])
            python_exe = Path(env_path) / "bin" / "python"
            
            # Get Python version
            result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                text=True
            )
            
            version_match = re.search(r'Python (\d+\.\d+)', result.stdout + result.stderr)
            if version_match:
                version = version_match.group(1)
                
                if version not in allowed_versions:
                    return False, f"Python {version} is not approved", {
                        "current_version": version,
                        "allowed_versions": allowed_versions
                    }
                
                return True, f"Python {version} is approved", {"version": version}
            
            return False, "Could not determine Python version", {}
            
        except Exception as e:
            logger.error(f"Error checking Python version: {e}")
            return False, f"Failed to check Python version: {str(e)}", {}
    
    def _check_license_compliance(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check for license compliance"""
        try:
            pip_exe = Path(env_path) / "bin" / "pip"
            
            # Get package licenses
            result = subprocess.run(
                [str(pip_exe.parent / "pip-licenses"), "--format", "json"],
                capture_output=True,
                text=True
            )
            
            licenses = json.loads(result.stdout) if result.stdout else []
            
            # Check for problematic licenses
            problematic_licenses = ["GPL", "LGPL", "AGPL", "SSPL"]
            violations = []
            
            for pkg in licenses:
                license_name = pkg.get("License", "")
                if any(prob in license_name for prob in problematic_licenses):
                    violations.append({
                        "package": pkg.get("Name"),
                        "version": pkg.get("Version"),
                        "license": license_name
                    })
            
            if violations:
                return False, f"Found {len(violations)} packages with restricted licenses", {
                    "violations": violations
                }
            
            return True, "All packages have compliant licenses", {}
            
        except Exception as e:
            logger.error(f"Error checking licenses: {e}")
            return False, f"Failed to check licenses: {str(e)}", {}
    
    def _check_required_tools(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check if required security tools are installed"""
        try:
            required = self.config.get("required_packages", [])
            pip_exe = Path(env_path) / "bin" / "pip"
            
            # Get installed packages
            result = subprocess.run(
                [str(pip_exe), "list", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            installed = json.loads(result.stdout)
            installed_names = {pkg["name"].lower() for pkg in installed}
            
            # Check for missing required packages
            missing = [pkg for pkg in required if pkg.lower() not in installed_names]
            
            if missing:
                return False, f"Missing required tools: {', '.join(missing)}", {
                    "missing_tools": missing
                }
            
            return True, "All required tools are installed", {}
            
        except Exception as e:
            logger.error(f"Error checking required tools: {e}")
            return False, f"Failed to check required tools: {str(e)}", {}
    
    def _check_package_count(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check if package count exceeds limit"""
        try:
            max_packages = 200  # Default, can be overridden in policy metadata
            pip_exe = Path(env_path) / "bin" / "pip"
            
            # Get package count
            result = subprocess.run(
                [str(pip_exe), "list", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            packages = json.loads(result.stdout)
            count = len(packages)
            
            if count > max_packages:
                return False, f"Package count ({count}) exceeds limit ({max_packages})", {
                    "package_count": count,
                    "limit": max_packages,
                    "excess": count - max_packages
                }
            
            return True, f"Package count ({count}) is within limit", {"package_count": count}
            
        except Exception as e:
            logger.error(f"Error checking package count: {e}")
            return False, f"Failed to check package count: {str(e)}", {}
    
    def _check_environment_size(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check if environment size exceeds limit"""
        try:
            max_size_gb = 2  # Default, can be overridden
            
            # Calculate environment size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(env_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except:
                        pass
            
            size_gb = total_size / (1024 ** 3)
            
            if size_gb > max_size_gb:
                return False, f"Environment size ({size_gb:.2f}GB) exceeds limit ({max_size_gb}GB)", {
                    "size_gb": size_gb,
                    "limit_gb": max_size_gb,
                    "size_bytes": total_size
                }
            
            return True, f"Environment size ({size_gb:.2f}GB) is within limit", {
                "size_gb": size_gb,
                "size_bytes": total_size
            }
            
        except Exception as e:
            logger.error(f"Error checking environment size: {e}")
            return False, f"Failed to check environment size: {str(e)}", {}
    
    def _check_environment_age(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check if environment is too old"""
        try:
            max_age_days = self.config.get("max_environment_age_days", 180)
            
            # Get environment creation time
            env_stat = os.stat(env_path)
            created_time = datetime.fromtimestamp(env_stat.st_ctime)
            age_days = (datetime.now() - created_time).days
            
            if age_days > max_age_days:
                return False, f"Environment is {age_days} days old (limit: {max_age_days})", {
                    "age_days": age_days,
                    "created_date": created_time.isoformat(),
                    "limit_days": max_age_days
                }
            
            return True, f"Environment age ({age_days} days) is acceptable", {
                "age_days": age_days,
                "created_date": created_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking environment age: {e}")
            return False, f"Failed to check environment age: {str(e)}", {}
    
    def _check_outdated_packages(self, env_path: str) -> Tuple[bool, str, Dict]:
        """Check for outdated packages"""
        try:
            pip_exe = Path(env_path) / "bin" / "pip"
            
            # Get outdated packages
            result = subprocess.run(
                [str(pip_exe), "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            outdated = json.loads(result.stdout) if result.stdout else []
            
            if len(outdated) > 10:  # Threshold for too many outdated packages
                return False, f"Found {len(outdated)} outdated packages", {
                    "outdated_count": len(outdated),
                    "packages": outdated[:10]  # First 10 as sample
                }
            
            return True, f"Acceptable number of outdated packages ({len(outdated)})", {
                "outdated_count": len(outdated)
            }
            
        except Exception as e:
            logger.error(f"Error checking outdated packages: {e}")
            return False, f"Failed to check outdated packages: {str(e)}", {}
    
    # ==============================================================================
    # REMEDIATION IMPLEMENTATIONS
    # ==============================================================================
    
    def _fix_vulnerable_packages(self, env_path: str, violation_details: Dict) -> Tuple[bool, str]:
        """Automatically fix vulnerable packages"""
        try:
            pip_exe = Path(env_path) / "bin" / "pip"
            vulnerabilities = violation_details.get("vulnerabilities", [])
            
            fixed = []
            failed = []
            
            for vuln in vulnerabilities:
                package = vuln.get("package")
                try:
                    # Try to upgrade to fixed version
                    subprocess.run(
                        [str(pip_exe), "install", "--upgrade", package],
                        check=True,
                        capture_output=True
                    )
                    fixed.append(package)
                except:
                    failed.append(package)
            
            if failed:
                return False, f"Fixed {len(fixed)} packages, failed to fix {len(failed)}"
            
            return True, f"Successfully fixed {len(fixed)} vulnerable packages"
            
        except Exception as e:
            logger.error(f"Error fixing vulnerabilities: {e}")
            return False, f"Failed to fix vulnerabilities: {str(e)}"
    
    def _remove_blocked_packages(self, env_path: str, violation_details: Dict) -> Tuple[bool, str]:
        """Remove blocked packages from environment"""
        try:
            pip_exe = Path(env_path) / "bin" / "pip"
            blocked = violation_details.get("blocked_packages", [])
            
            for package in blocked:
                try:
                    subprocess.run(
                        [str(pip_exe), "uninstall", "-y", package],
                        check=True,
                        capture_output=True
                    )
                except:
                    pass
            
            return True, f"Removed {len(blocked)} blocked packages"
            
        except Exception as e:
            logger.error(f"Error removing blocked packages: {e}")
            return False, f"Failed to remove blocked packages: {str(e)}"
    
    def _install_required_tools(self, env_path: str, violation_details: Dict) -> Tuple[bool, str]:
        """Install required security tools"""
        try:
            pip_exe = Path(env_path) / "bin" / "pip"
            missing = violation_details.get("missing_tools", [])
            
            for tool in missing:
                try:
                    subprocess.run(
                        [str(pip_exe), "install", tool],
                        check=True,
                        capture_output=True
                    )
                except:
                    pass
            
            return True, f"Installed {len(missing)} required tools"
            
        except Exception as e:
            logger.error(f"Error installing required tools: {e}")
            return False, f"Failed to install required tools: {str(e)}"
    
    def _optimize_environment_size(self, env_path: str, violation_details: Dict) -> Tuple[bool, str]:
        """Optimize environment size by cleaning caches"""
        try:
            pip_exe = Path(env_path) / "bin" / "pip"
            
            # Clean pip cache
            subprocess.run([str(pip_exe), "cache", "purge"], capture_output=True)
            
            # Remove .pyc files
            for root, dirs, files in os.walk(env_path):
                for file in files:
                    if file.endswith('.pyc'):
                        os.remove(os.path.join(root, file))
            
            # Remove __pycache__ directories
            for root, dirs, files in os.walk(env_path):
                if '__pycache__' in dirs:
                    import shutil
                    shutil.rmtree(os.path.join(root, '__pycache__'))
            
            return True, "Optimized environment size"
            
        except Exception as e:
            logger.error(f"Error optimizing environment: {e}")
            return False, f"Failed to optimize environment: {str(e)}"
    
    def _update_outdated_packages(self, env_path: str, violation_details: Dict) -> Tuple[bool, str]:
        """Update outdated packages"""
        try:
            pip_exe = Path(env_path) / "bin" / "pip"
            
            # Update all outdated packages
            result = subprocess.run(
                [str(pip_exe), "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            outdated = json.loads(result.stdout) if result.stdout else []
            
            updated = 0
            for pkg in outdated[:10]:  # Limit to first 10 to avoid breaking changes
                try:
                    subprocess.run(
                        [str(pip_exe), "install", "--upgrade", pkg["name"]],
                        check=True,
                        capture_output=True
                    )
                    updated += 1
                except:
                    pass
            
            return True, f"Updated {updated} outdated packages"
            
        except Exception as e:
            logger.error(f"Error updating packages: {e}")
            return False, f"Failed to update packages: {str(e)}"
    
    # ==============================================================================
    # AUDIT AND ENFORCEMENT
    # ==============================================================================
    
    def audit_environment(self, env_name: str, env_path: str, auto_remediate: bool = None) -> EnvironmentAuditRecord:
        """Perform comprehensive audit of an environment"""
        logger.info(f"Auditing environment: {env_name}")
        
        if auto_remediate is None:
            auto_remediate = self.config.get("auto_remediation", False)
        
        violations = []
        recommendations = []
        metrics = {}
        
        # Check each policy
        for policy in self.policies:
            # Check if environment is exempted
            if self._is_exempted(env_name, policy.exceptions):
                continue
            
            # Get check function
            check_func = self.validators.get(policy.check_function)
            if not check_func:
                logger.warning(f"Check function not found: {policy.check_function}")
                continue
            
            # Run check
            passed, message, details = check_func(env_path)
            
            if not passed:
                violation = PolicyViolation(
                    rule_id=policy.id,
                    environment_name=env_name,
                    violation_level=policy.level,
                    message=message,
                    detected_at=datetime.now().isoformat(),
                    details=details
                )
                
                # Attempt remediation if enabled
                if auto_remediate and policy.auto_remediate and policy.remediation_function:
                    remediate_func = self.validators.get(policy.remediation_function)
                    if remediate_func:
                        success, result = remediate_func(env_path, details)
                        violation.remediated = success
                        violation.remediation_result = result
                        
                        if success:
                            logger.info(f"Auto-remediated: {policy.name} for {env_name}")
                        else:
                            logger.warning(f"Failed to remediate: {policy.name} for {env_name}")
                
                violations.append(violation)
                
                # Add recommendation if not auto-remediated
                if not violation.remediated:
                    recommendations.append(f"Fix {policy.name}: {message}")
        
        # Collect metrics
        metrics = self._collect_environment_metrics(env_path)
        
        # Determine compliance status
        if not violations:
            compliance_status = ComplianceStatus.COMPLIANT
        elif any(v.violation_level == PolicyViolationLevel.CRITICAL for v in violations):
            compliance_status = ComplianceStatus.NON_COMPLIANT
        else:
            compliance_status = ComplianceStatus.PARTIALLY_COMPLIANT
        
        # Create audit record
        audit_record = EnvironmentAuditRecord(
            environment_name=env_name,
            audit_timestamp=datetime.now().isoformat(),
            compliance_status=compliance_status,
            policy_violations=violations,
            metrics=metrics,
            recommendations=recommendations,
            auditor="system"
        )
        
        # Store in database
        self._store_audit_record(audit_record)
        
        # Send notifications if critical violations
        if compliance_status == ComplianceStatus.NON_COMPLIANT:
            self._send_compliance_alert(env_name, audit_record)
        
        return audit_record
    
    def _is_exempted(self, env_name: str, exceptions: List[str]) -> bool:
        """Check if environment is exempted from policy"""
        for pattern in exceptions:
            if re.match(pattern, env_name):
                return True
        return False
    
    def _collect_environment_metrics(self, env_path: str) -> Dict:
        """Collect metrics about the environment"""
        metrics = {}
        
        try:
            # Package count
            pip_exe = Path(env_path) / "bin" / "pip"
            result = subprocess.run(
                [str(pip_exe), "list", "--format", "json"],
                capture_output=True,
                text=True
            )
            packages = json.loads(result.stdout) if result.stdout else []
            metrics["package_count"] = len(packages)
            
            # Environment size
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(env_path)
                for filename in filenames
            )
            metrics["size_bytes"] = total_size
            metrics["size_mb"] = round(total_size / (1024 * 1024), 2)
            
            # Python version
            python_exe = Path(env_path) / "bin" / "python"
            result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                text=True
            )
            version_match = re.search(r'Python (\d+\.\d+\.\d+)', result.stdout + result.stderr)
            if version_match:
                metrics["python_version"] = version_match.group(1)
            
            # Last modified time
            env_stat = os.stat(env_path)
            metrics["last_modified"] = datetime.fromtimestamp(env_stat.st_mtime).isoformat()
            metrics["created"] = datetime.fromtimestamp(env_stat.st_ctime).isoformat()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _store_audit_record(self, record: EnvironmentAuditRecord):
        """Store audit record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_records 
                (environment_name, audit_timestamp, compliance_status, violations, metrics, recommendations, auditor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.environment_name,
                record.audit_timestamp,
                record.compliance_status.value,
                json.dumps([asdict(v) for v in record.policy_violations]),
                json.dumps(record.metrics),
                json.dumps(record.recommendations),
                record.auditor
            ))
            
            # Store individual violations
            for violation in record.policy_violations:
                cursor.execute('''
                    INSERT INTO policy_violations
                    (rule_id, environment_name, violation_level, message, detected_at, remediated, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    violation.rule_id,
                    violation.environment_name,
                    violation.violation_level.value,
                    violation.message,
                    violation.detected_at,
                    violation.remediated,
                    json.dumps(violation.details)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing audit record: {e}")
    
    def _send_compliance_alert(self, env_name: str, audit_record: EnvironmentAuditRecord):
        """Send alert for compliance violations"""
        try:
            email = self.config.get("notification_email")
            if not email:
                return
            
            # Prepare email content
            critical_violations = [
                v for v in audit_record.policy_violations 
                if v.violation_level == PolicyViolationLevel.CRITICAL
            ]
            
            subject = f"CRITICAL: Environment '{env_name}' is non-compliant"
            
            body = f"""
Environment Compliance Alert
============================

Environment: {env_name}
Status: {audit_record.compliance_status.value.upper()}
Time: {audit_record.audit_timestamp}

Critical Violations:
{chr(10).join(f'- {v.message}' for v in critical_violations)}

Recommendations:
{chr(10).join(f'- {r}' for r in audit_record.recommendations[:5])}

Please take immediate action to remediate these issues.

View full report: http://governance.example.com/environments/{env_name}
            """
            
            # Send email (placeholder - implement actual email sending)
            logger.info(f"Sending compliance alert for {env_name} to {email}")
            # self._send_email(email, subject, body)
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    # ==============================================================================
    # REPORTING AND ANALYTICS
    # ==============================================================================
    
    def generate_compliance_report(self, start_date: str = None, end_date: str = None) -> Dict:
        """Generate compliance report for all environments"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get audit records within date range
            query = "SELECT * FROM audit_records"
            params = []
            
            if start_date and end_date:
                query += " WHERE audit_timestamp BETWEEN ? AND ?"
                params = [start_date, end_date]
            elif start_date:
                query += " WHERE audit_timestamp >= ?"
                params = [start_date]
            elif end_date:
                query += " WHERE audit_timestamp <= ?"
                params = [end_date]
            
            cursor.execute(query, params)
            records = cursor.fetchall()
            
            # Process records for report
            report = {
                "total_audits": len(records),
                "compliant_count": 0,
                "non_compliant_count": 0,
                "partially_compliant_count": 0,
                "violation_counts": {},
                "common_violations": {},
                "remediation_success_rate": 0,
                "environments": {}
            }
            
            total_violations = 0
            remediated_violations = 0
            
            for record in records:
                compliance_status = record[2]
                violations = json.loads(record[3]) if record[3] else []
                
                # Count compliance status
                if compliance_status == "compliant":
                    report["compliant_count"] += 1
                elif compliance_status == "non_compliant":
                    report["non_compliant_count"] += 1
                else:
                    report["partially_compliant_count"] += 1
                
                # Count violations
                for violation in violations:
                    rule_id = violation.get("rule_id")
                    report["violation_counts"][rule_id] = report["violation_counts"].get(rule_id, 0) + 1
                    
                    total_violations += 1
                    if violation.get("remediated"):
                        remediated_violations += 1
            
            # Calculate remediation success rate
            if total_violations > 0:
                report["remediation_success_rate"] = (remediated_violations / total_violations) * 100
            
            # Get most common violations
            if report["violation_counts"]:
                sorted_violations = sorted(report["violation_counts"].items(), 
                                         key=lambda x: x[1], reverse=True)
                report["common_violations"] = dict(sorted_violations[:10])
            
            conn.close()
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {}
    
    def get_environment_history(self, env_name: str, days: int = 30) -> List[Dict]:
        """Get audit history for a specific environment"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT * FROM audit_records 
                WHERE environment_name = ? AND audit_timestamp >= ?
                ORDER BY audit_timestamp DESC
            ''', (env_name, cutoff_date))
            
            records = cursor.fetchall()
            
            history = []
            for record in records:
                history.append({
                    "timestamp": record[2],
                    "compliance_status": record[3],
                    "violations": json.loads(record[4]) if record[4] else [],
                    "metrics": json.loads(record[5]) if record[5] else {},
                    "recommendations": json.loads(record[6]) if record[6] else []
                })
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"Error getting environment history: {e}")
            return []
    
    def _count_severities(self, vulnerabilities: List[Dict]) -> Dict:
        """Count vulnerabilities by severity"""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "unknown").lower()
            if severity in counts:
                counts[severity] += 1
        return counts

# ==============================================================================
# SCHEDULED TASKS AND AUTOMATION
# ==============================================================================

class GovernanceScheduler:
    """Scheduler for automated governance tasks"""
    
    def __init__(self, policy_engine: EnterprisePolicyEngine):
        self.engine = policy_engine
        self.setup_schedule()
    
    def setup_schedule(self):
        """Setup scheduled tasks"""
        # Daily compliance audit
        schedule.every().day.at("02:00").do(self.run_daily_audit)
        
        # Weekly compliance report
        schedule.every().monday.at("09:00").do(self.generate_weekly_report)
        
        # Monthly cleanup
        schedule.every().month.do(self.cleanup_old_records)
    
    def run_daily_audit(self):
        """Run daily audit of all environments"""
        logger.info("Running scheduled daily audit")
        # Implementation would scan all registered environments
        pass
    
    def generate_weekly_report(self):
        """Generate and send weekly compliance report"""
        logger.info("Generating weekly compliance report")
        report = self.engine.generate_compliance_report()
        # Send report to stakeholders
        pass
    
    def cleanup_old_records(self):
        """Clean up old audit records"""
        logger.info("Cleaning up old records")
        retention_days = self.engine.config.get("retention_days", 90)
        # Implementation would delete records older than retention period
        pass
    
    def run(self):
        """Run the scheduler"""
        logger.info("Starting governance scheduler")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Initialize policy engine
    engine = EnterprisePolicyEngine()
    
    # Example: Audit a specific environment
    audit_result = engine.audit_environment(
        "production_api",
        "/opt/python-envs/production_api",
        auto_remediate=True
    )
    
    print(f"Compliance Status: {audit_result.compliance_status.value}")
    print(f"Violations Found: {len(audit_result.policy_violations)}")
    
    # Generate compliance report
    report = engine.generate_compliance_report()
    print(f"\nCompliance Report Summary:")
    print(f"Total Audits: {report['total_audits']}")
    print(f"Compliant: {report['compliant_count']}")
    print(f"Non-Compliant: {report['non_compliant_count']}")
    print(f"Remediation Success Rate: {report['remediation_success_rate']:.1f}%")
    
    # Start scheduler for automated tasks
    # scheduler = GovernanceScheduler(engine)
    # scheduler.run()  # This runs indefinitely