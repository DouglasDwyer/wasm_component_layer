use std::sync::Arc;

use anyhow::{Error, Ok, Result};

#[cfg(feature = "serde")]
use serde::*;

use wit_parser::validate_id;

/// Describes the name of a component type.
#[derive(Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TypeIdentifier {
    /// The name of the type.
    name: Arc<str>,
    /// The interface in which the type was defined, if any.
    interface: Option<InterfaceIdentifier>,
}

impl TypeIdentifier {
    /// Creates a new type identifier for the given name and interface.
    pub fn new(name: impl Into<Arc<str>>, interface: Option<InterfaceIdentifier>) -> Self {
        Self {
            name: name.into(),
            interface,
        }
    }

    /// Gets the name of the type.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the interface in which the type was defined, if any.
    pub fn interface(&self) -> Option<&InterfaceIdentifier> {
        self.interface.as_ref()
    }
}

impl std::fmt::Debug for TypeIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for TypeIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(inter) = &self.interface {
            f.write_fmt(format_args!("{}.{}", inter, self.name()))
        } else {
            f.write_fmt(format_args!("{}", self.name()))
        }
    }
}

/// Uniquely identifies a WASM package within a registry.
#[derive(Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PackageName {
    /// The namespace of the package.
    namespace: Arc<str>,
    /// The name of the package.
    name: Arc<str>,
}

impl PackageName {
    /// Creates a new package identifier for the given namespace and name.
    pub fn new(namespace: impl Into<Arc<str>>, name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            namespace: namespace.into(),
        }
    }

    /// The name of the package.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The namespace of the package.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
}

impl TryFrom<&str> for PackageName {
    type Error = Error;

    fn try_from(id: &str) -> Result<Self> {
        let colon = id
            .find(':')
            .ok_or_else(|| Error::msg(format!("Expected ':' in identifier {id}")))?;
        let colon_next = colon + 1;
        let namespace = &id[0..colon];
        validate_id(namespace)?;
        let name = &id[colon_next..];
        validate_id(name)?;

        Ok(Self::new(namespace, name))
    }
}

impl std::fmt::Debug for PackageName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for PackageName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}:{}", self.namespace(), self.name()))
    }
}

/// Uniquely identifies a WASM package within a registry, with an optionally-associated version.
#[derive(Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PackageIdentifier {
    /// The name of the package.
    name: PackageName,
    /// The version of the package.
    version: Option<semver::Version>,
}

impl PackageIdentifier {
    /// Creates a new package identifier for the given namespace, name, and version.
    pub fn new(name: PackageName, version: Option<semver::Version>) -> Self {
        Self { name, version }
    }

    /// Gets the name of the package.
    pub fn name(&self) -> &PackageName {
        &self.name
    }

    /// Gets the version of the package, if any.
    pub fn version(&self) -> Option<&semver::Version> {
        self.version.as_ref()
    }
}

impl From<&wit_parser::PackageName> for PackageIdentifier {
    fn from(value: &wit_parser::PackageName) -> Self {
        Self {
            name: PackageName::new(value.namespace.as_str(), value.name.as_str()),
            version: value.version.clone(),
        }
    }
}

impl TryFrom<&str> for PackageIdentifier {
    type Error = Error;

    fn try_from(id: &str) -> Result<Self> {
        let colon = id
            .find(':')
            .ok_or_else(|| Error::msg(format!("Expected ':' in identifier {id}")))?;
        let colon_next = colon + 1;
        let namespace = &id[0..colon];
        validate_id(namespace)?;
        let version_index = id[colon_next..].find('@').map(|x| x + colon_next);
        let name = &id[colon_next..version_index.unwrap_or(id.len())];
        validate_id(name)?;

        let version = if let Some(idx) = version_index {
            Some(semver::Version::parse(&id[(idx + 1)..])?)
        } else {
            None
        };

        Ok(Self {
            name: PackageName::new(namespace, name),
            version,
        })
    }
}

impl std::fmt::Debug for PackageIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for PackageIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(version) = self.version() {
            f.write_fmt(format_args!(
                "{}:{}@{}",
                self.name().namespace(),
                self.name().name(),
                version
            ))
        } else {
            f.write_fmt(format_args!(
                "{}:{}",
                self.name().namespace(),
                self.name().name()
            ))
        }
    }
}

/// Uniquely identifies a component model interface within a package.
#[derive(Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InterfaceIdentifier {
    /// The package ID.
    package: PackageIdentifier,
    /// The name of the interface.
    name: Arc<str>,
}

impl InterfaceIdentifier {
    /// Creates a new interface identifier for the given name and package.
    pub fn new(package: PackageIdentifier, name: impl Into<Arc<str>>) -> Self {
        Self {
            package,
            name: name.into(),
        }
    }

    /// Gets the identifier of the package.
    pub fn package(&self) -> &PackageIdentifier {
        &self.package
    }

    /// Gets the name of the interface.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl TryFrom<&str> for InterfaceIdentifier {
    type Error = Error;

    fn try_from(id: &str) -> Result<Self> {
        let colon = id
            .find(':')
            .ok_or_else(|| Error::msg(format!("Expected ':' in identifier {id}")))?;
        let colon_next = colon + 1;
        let namespace = &id[0..colon];
        validate_id(namespace)?;
        let interface_index = id[colon_next..]
            .find('/')
            .ok_or_else(|| Error::msg(format!("Expected '/' in identifier {id}")))?
            + colon_next;
        let interface_index_next = interface_index + 1;

        let name = &id[(colon + 1)..interface_index];
        validate_id(name)?;

        let version_index = id[interface_index_next..]
            .find('@')
            .map(|x| x + interface_index_next);
        let interface_name = &id[interface_index_next..version_index.unwrap_or(id.len())];
        validate_id(interface_name)?;

        let version = if let Some(idx) = version_index {
            Some(semver::Version::parse(&id[(idx + 1)..])?)
        } else {
            None
        };

        Ok(Self {
            package: PackageIdentifier::new(PackageName::new(namespace, name), version),
            name: interface_name.into(),
        })
    }
}

impl std::fmt::Debug for InterfaceIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for InterfaceIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(version) = self.package.version() {
            f.write_fmt(format_args!(
                "{}:{}/{}@{}",
                self.package().name().namespace(),
                self.package().name().name(),
                self.name(),
                version
            ))
        } else {
            f.write_fmt(format_args!(
                "{}:{}/{}",
                self.package().name().namespace(),
                self.package().name().name(),
                self.name()
            ))
        }
    }
}
